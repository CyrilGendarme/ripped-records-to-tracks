"""
Discogs search via the REST API directly (httpx).

Matches the documented curl:
  GET https://api.discogs.com/database/search
      ?artist=Radiohead&release_title=OK+Computer&type=release&token=YOUR_TOKEN

Authentication priority:
  1. DISCOGS_TOKEN  – personal access token (simplest, 60 req/min)
  2. DISCOGS_CONSUMER_KEY + DISCOGS_CONSUMER_SECRET  (25 req/min)
  3. No auth – 25 req/min

Get a personal token at: https://www.discogs.com/settings/developers
"""

from typing import List, Optional
import logging
import re

import httpx

from .config import settings
from .discogs_models import (
    DiscogsReleaseInfo,
    DiscogsReleaseTrack,
    DiscogsResult,
    DiscogsTrack,
    PriceStats,
)

logger = logging.getLogger(__name__)

_BASE = "https://api.discogs.com"
_HEADERS = {
    "User-Agent": "DiggerHelper/1.0 +https://github.com/CyrilGendarme/digger-helper"
}
_CATNO_NORMALIZE_RE = re.compile(r"[^A-Z0-9]")
_SIDE_RE = re.compile(r"^\s*([A-D])")


def _auth_params() -> dict:
    """Return the best available auth query params."""
    if settings.discogs_token:
        return {"token": settings.discogs_token}
    if settings.discogs_consumer_key and settings.discogs_consumer_secret:
        return {
            "key": settings.discogs_consumer_key,
            "secret": settings.discogs_consumer_secret,
        }
    return {}


def _safe(val) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    return s or None


def _normalize_catno(value: Optional[str]) -> str:
    if not value:
        return ""
    return _CATNO_NORMALIZE_RE.sub("", value.upper())


def _extract_side(position: Optional[str]) -> Optional[str]:
    if not position:
        return None
    match = _SIDE_RE.match(position.upper())
    if not match:
        return None
    return match.group(1)


def _search_raw(params: dict, limit: int) -> list:
    """Hit /database/search and return up to `limit` result dicts."""
    query = {
        "type": "release",
        "per_page": limit,
        "page": 1,
        **_auth_params(),
        **params,
    }
    resp = httpx.get(
        f"{_BASE}/database/search", params=query, headers=_HEADERS, timeout=15
    )
    resp.raise_for_status()
    return resp.json().get("results", [])[:limit]


def _fetch_release(release_id: int) -> dict:
    """Fetch full release detail from /releases/{id}."""
    resp = httpx.get(
        f"{_BASE}/releases/{release_id}",
        params=_auth_params(),
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_price_stats(release_id: int) -> Optional[PriceStats]:
    """Fetch num_for_sale and lowest_price from /marketplace/stats/{id}."""
    try:
        resp = httpx.get(
            f"{_BASE}/marketplace/stats/{release_id}",
            params={**_auth_params(), "curr_abbr": "EUR"},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        data: dict = resp.json()
    except Exception as exc:
        logger.warning("Marketplace stats unavailable for %s: %s", release_id, exc)
        return None

    num_for_sale = data.get("num_for_sale")
    lowest_raw = data.get("lowest_price")
    if isinstance(lowest_raw, dict):
        lowest = (
            float(lowest_raw["value"]) if lowest_raw.get("value") is not None else None
        )
        currency = lowest_raw.get("currency")
    elif lowest_raw is not None:
        lowest = float(lowest_raw)
        currency = None
    else:
        lowest = None
        currency = None

    if num_for_sale is None and lowest is None:
        return None

    return PriceStats(
        currency=currency,
        num_for_sale=int(num_for_sale) if num_for_sale is not None else None,
        lowest=lowest,
    )


def _parse_result(item: dict) -> Optional[DiscogsResult]:
    """Convert a search result item + full release detail into a DiscogsResult."""
    release_id = item.get("id")
    if not release_id:
        return None

    try:
        detail = _fetch_release(release_id)
    except Exception as exc:
        logger.warning("Could not fetch release %s detail: %s", release_id, exc)
        detail = item  # fall back to search result data

    # Tracklist
    tracklist = [
        DiscogsTrack(
            position=_safe(t.get("position")),
            title=t.get("title", ""),
            duration=_safe(t.get("duration")),
        )
        for t in (detail.get("tracklist") or [])
    ]

    # Label / catno
    labels = detail.get("labels") or item.get("label") or []
    if isinstance(labels, list) and labels and isinstance(labels[0], dict):
        label_name = _safe(labels[0].get("name"))
        catno_val = _safe(labels[0].get("catno"))
    elif isinstance(labels, list) and labels and isinstance(labels[0], str):
        label_name = labels[0]
        catno_val = _safe(item.get("catno"))
    else:
        label_name = None
        catno_val = _safe(item.get("catno"))

    # Format
    formats = detail.get("formats") or item.get("formats") or []
    fmt = (
        _safe(formats[0].get("name"))
        if formats and isinstance(formats[0], dict)
        else None
    )

    # Artist
    artists = detail.get("artists") or []
    if artists:
        artist_str = (
            " & ".join(
                a.get("name", "") for a in artists if isinstance(a, dict)
            ).strip()
            or None
        )
    else:
        raw_artist = item.get("artist") or item.get("artists") or None
        if isinstance(raw_artist, list):
            artist_str = " & ".join(raw_artist)
        else:
            artist_str = _safe(raw_artist)

    title = _safe(detail.get("title") or item.get("title")) or ""

    # Marketplace pricing (API only exposes num_for_sale + lowest_price)
    price_stats = _fetch_price_stats(release_id)

    return DiscogsResult(
        id=release_id,
        title=title,
        artist=artist_str,
        year=_safe(str(detail.get("year") or item.get("year") or "")),
        label=label_name,
        catno=catno_val,
        format=fmt,
        cover_image=_safe(item.get("cover_image") or item.get("thumb")),
        resource_url=f"https://www.discogs.com/release/{release_id}",
        tracklist=tracklist,
        price_stats=price_stats,
    )


def search_releases(
    artist: Optional[str] = None,
    track: Optional[str] = None,
    album: Optional[str] = None,
    catno: Optional[str] = None,
    limit: int = 5,
) -> List[DiscogsResult]:
    # ── Primary: field-specific search ───────────────────────────────────────
    params: dict = {}
    if artist:
        params["artist"] = artist
    if track:
        params["track"] = track
    if album:
        params["release_title"] = album
    if catno:
        params["catno"] = catno

    raw = _search_raw(params, limit)
    logger.info("Discogs field-search returned %d results", len(raw))

    # ── Fallback: free-text q= (tolerates OCR capitalisation noise) ──────────
    if not raw:
        q = " ".join(
            p.strip() for p in [artist, album, track, catno] if p and p.strip()
        )
        if q:
            logger.info("Retrying Discogs with q=%r", q)
            raw = _search_raw({"q": q}, limit)
            logger.info("Discogs q-search returned %d results", len(raw))

    output: List[DiscogsResult] = []
    for item in raw:
        result = _parse_result(item)
        if result:
            output.append(result)

    return output


def _search_releases_by_record_ref(
    record_ref: str,
    artist: Optional[str],
    limit: int,
) -> List[DiscogsResult]:
    releases = search_releases(artist=artist, catno=record_ref, limit=limit)
    if releases:
        return releases

    fallback_query = {"q": f"{artist or ''} {record_ref}".strip()}
    raw = _search_raw(fallback_query, limit)
    fallback: List[DiscogsResult] = []
    for item in raw:
        parsed = _parse_result(item)
        if parsed:
            fallback.append(parsed)
    return fallback


def _pick_release_for_record_ref(
    releases: List[DiscogsResult],
    record_ref: str,
) -> Optional[DiscogsResult]:
    if not releases:
        return None
    wanted = _normalize_catno(record_ref)
    for release in releases:
        if _normalize_catno(release.catno) == wanted:
            return release
    return releases[0]


def get_release_info_by_record_ref(
    record_ref: str,
    artist: Optional[str] = None,
    limit: int = 8,
) -> Optional[DiscogsReleaseInfo]:
    """Resolve record reference to release metadata and A/B/C/D tracks."""
    record_ref = (record_ref or "").strip()
    if not record_ref:
        return None

    releases = _search_releases_by_record_ref(
        record_ref=record_ref,
        artist=artist,
        limit=limit,
    )
    chosen = _pick_release_for_record_ref(releases, record_ref=record_ref)
    if chosen is None:
        return None

    tracks: List[DiscogsReleaseTrack] = []
    for track in chosen.tracklist:
        position = _safe(track.position)
        side = _extract_side(position)
        title = _safe(track.title)
        if not title:
            continue
        tracks.append(
            DiscogsReleaseTrack(
                side=side or "",
                position=position or "",
                title=title,
                duration=_safe(track.duration),
            )
        )

    return DiscogsReleaseInfo(
        artist=chosen.artist,
        album=_safe(chosen.title),
        record_ref=_safe(chosen.catno) or record_ref,
        tracks=tracks,
    )


def get_release_tracks_by_record_ref(
    record_ref: str,
    artist: Optional[str] = None,
    limit: int = 8,
) -> List[DiscogsReleaseTrack]:
    """Resolve a record reference (catalog number) to release tracks with A/B/C/D sides.

    Returns only tracks whose `position` starts with side A, B, C or D.
    """
    record_ref = (record_ref or "").strip()
    if not record_ref:
        return []

    info = get_release_info_by_record_ref(record_ref=record_ref, artist=artist, limit=limit)
    if info is None:
        return []
    return info.tracks
