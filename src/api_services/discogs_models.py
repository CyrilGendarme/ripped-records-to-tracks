from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from enum import Enum


class FieldType(str, Enum):
    track_name = "track_name"
    artist_name = "artist_name"
    album_name = "album_name"
    record_ref = "record_ref"
    unknown = "unknown"


class LabelledField(BaseModel):
    text: str
    field_type: FieldType


class RecordInfo(BaseModel):
    track_name: Optional[str] = None
    artist_name: Optional[str] = None
    album_name: Optional[str] = None
    record_ref: Optional[str] = None


# ── Discogs ────────────────────────────────────────────────────────────────
class DiscogsTrack(BaseModel):
    position: Optional[str] = None
    title: str
    duration: Optional[str] = None


class DiscogsReleaseTrack(BaseModel):
    side: str
    position: str
    title: str
    duration: Optional[str] = None


class DiscogsReleaseInfo(BaseModel):
    artist: Optional[str] = None
    album: Optional[str] = None
    record_ref: Optional[str] = None
    tracks: List[DiscogsReleaseTrack] = []


class PriceStats(BaseModel):
    currency: Optional[str] = None
    num_for_sale: Optional[int] = None
    lowest: Optional[float] = None


class DiscogsResult(BaseModel):
    id: int
    title: str
    artist: Optional[str] = None
    year: Optional[str] = None
    label: Optional[str] = None
    catno: Optional[str] = None
    format: Optional[str] = None
    cover_image: Optional[str] = None
    resource_url: str
    tracklist: List[DiscogsTrack] = []
    price_stats: Optional[PriceStats] = None


class DiscogsSearchResponse(BaseModel):
    results: List[DiscogsResult]
    total: int


# ── Media search ──────────────────────────────────────────────────────────
class Platform(str, Enum):
    youtube = "youtube"
    soundcloud = "soundcloud"
    bandcamp = "bandcamp"


class MediaLink(BaseModel):
    platform: Platform
    title: str
    url: str
    thumbnail: Optional[str] = None
    duration: Optional[str] = None
    channel: Optional[str] = None
    price: Optional[str] = None


class MediaSearchResponse(BaseModel):
    links: List[MediaLink]
