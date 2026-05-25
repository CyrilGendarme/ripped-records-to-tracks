from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.rtt.pipeline import split_audio_file
from src.rtt.segmentation import SegmentationConfig


class DesktopApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Ripped Records to Tracks")
        self.root.geometry("1080x760")
        self.root.configure(bg="#f6f2e9")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.cwd() / "output_tracks"))
        self.status_var = tk.StringVar(value="Select an audio file to start.")

        self.min_track = tk.DoubleVar(value=40.0)
        self.silence_db = tk.DoubleVar(value=-36.0)
        self.silence_min = tk.DoubleVar(value=2)
        self.music_low_hz = tk.DoubleVar(value=120.0)
        self.music_high_hz = tk.DoubleVar(value=5000.0)
        self.trim_silence_db = tk.DoubleVar(value=-52.0)

        self._build_styles()
        self._build_layout()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Card.TLabelframe", background="#fffdf7", borderwidth=1)
        style.configure(
            "Card.TLabelframe.Label",
            background="#fffdf7",
            foreground="#1f2937",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("Main.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure(
            "TLabel", background="#f6f2e9", foreground="#111827", font=("Segoe UI", 10)
        )

    def _build_layout(self) -> None:
        wrapper = ttk.Frame(self.root, padding=14)
        wrapper.pack(fill="both", expand=True)

        title = ttk.Label(
            wrapper,
            text="Ripped Records to Tracks - Desktop",
            font=("Segoe UI", 17, "bold"),
        )
        title.pack(anchor="w", pady=(0, 8))

        picker = ttk.LabelFrame(
            wrapper, text="Audio & Output", style="Card.TLabelframe", padding=12
        )
        picker.pack(fill="x", pady=(0, 10))

        self._path_row(
            picker, "Input audio", self.input_path, self._pick_input, "Choose File", 0
        )
        self._path_row(
            picker,
            "Output folder",
            self.output_dir,
            self._pick_output,
            "Choose Folder",
            1,
        )

        body = ttk.Frame(wrapper)
        body.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(
            body, text="Segmentation Rules", style="Card.TLabelframe", padding=12
        )
        controls.pack(side="left", fill="y", padx=(0, 10))

        self._slider(controls, "Min track length (s)", self.min_track, 10.0, 240.0, 0)
        self._slider(
            controls, "Silence threshold (dB)", self.silence_db, -70.0, -10.0, 1
        )
        self._slider(controls, "Min silence window (s)", self.silence_min, 0.2, 6.0, 2)
        self._slider(controls, "Music low freq (Hz)", self.music_low_hz, 40.0, 800.0, 3)
        self._slider(
            controls, "Music high freq (Hz)", self.music_high_hz, 2000.0, 12000.0, 4
        )
        self._slider(
            controls,
            "Trim start/end silence (dB)",
            self.trim_silence_db,
            -80.0,
            -20.0,
            5,
        )

        run_btn = ttk.Button(
            controls,
            text="Split Into Tracks",
            style="Main.TButton",
            command=self._start_split,
        )
        run_btn.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(14, 4))

        self.open_btn = ttk.Button(
            controls, text="Open Output Folder", command=self._open_output_dir
        )
        self.open_btn.grid(row=7, column=0, columnspan=2, sticky="ew")

        result = ttk.LabelFrame(
            body, text="Detected Timeline", style="Card.TLabelframe", padding=10
        )
        result.pack(side="left", fill="both", expand=True)

        columns = ("track", "start", "end", "duration")
        self.tree = ttk.Treeview(result, columns=columns, show="headings", height=20)
        self.tree.heading("track", text="#")
        self.tree.heading("start", text="Start (s)")
        self.tree.heading("end", text="End (s)")
        self.tree.heading("duration", text="Duration (s)")

        self.tree.column("track", width=60, anchor="center")
        self.tree.column("start", width=110, anchor="e")
        self.tree.column("end", width=110, anchor="e")
        self.tree.column("duration", width=120, anchor="e")
        self.tree.pack(fill="both", expand=True)

        status = ttk.Label(wrapper, textvariable=self.status_var, font=("Consolas", 10))
        status.pack(anchor="w", pady=(10, 0))

    def _path_row(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.StringVar,
        callback,
        button_text: str,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        entry = ttk.Entry(parent, textvariable=var, width=92)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        btn = ttk.Button(parent, text=button_text, command=callback)
        btn.grid(row=row, column=2, padx=(8, 0), pady=4)
        parent.columnconfigure(1, weight=1)

    def _slider(
        self,
        parent: ttk.LabelFrame,
        label: str,
        var: tk.DoubleVar,
        frm: float,
        to: float,
        row: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        scale = ttk.Scale(parent, variable=var, from_=frm, to=to)
        scale.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=4)
        parent.columnconfigure(1, weight=1)

    def _pick_input(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.input_path.set(selected)

    def _pick_output(self) -> None:
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_dir.set(selected)

    def _open_output_dir(self) -> None:
        path = Path(self.output_dir.get())
        if path.exists():
            os.startfile(path)
        else:
            messagebox.showwarning(
                "Missing folder", "Output folder does not exist yet."
            )

    def _start_split(self) -> None:
        input_file = Path(self.input_path.get())
        output_folder = Path(self.output_dir.get())

        if not input_file.exists():
            messagebox.showerror(
                "Input missing", "Please choose a valid input audio file."
            )
            return

        self.status_var.set(
            "Analyzing audio... this can take a while for long recordings."
        )

        worker = threading.Thread(
            target=self._run_split, args=(input_file, output_folder), daemon=True
        )
        worker.start()

    def _run_split(self, input_file: Path, output_folder: Path) -> None:
        try:
            cfg = SegmentationConfig(
                min_track_len_s=float(self.min_track.get()),
                silence_db_threshold=float(self.silence_db.get()),
                silence_min_len_s=float(self.silence_min.get()),
                music_low_hz=float(self.music_low_hz.get()),
                music_high_hz=float(self.music_high_hz.get()),
                trim_silence_db_threshold=float(self.trim_silence_db.get()),
            )

            result = split_audio_file(
                file_path=input_file, output_dir=output_folder, cfg=cfg
            )
            self.root.after(
                0,
                lambda: self._render_result(result.segmentation.boundaries_s),
            )
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda: messagebox.showerror("Split failed", str(exc)))
            self.root.after(
                0,
                lambda: self.status_var.set(
                    "Split failed. Adjust parameters and retry."
                ),
            )

    def _render_result(self, boundaries: list[float]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx in range(len(boundaries) - 1):
            start_s = boundaries[idx]
            end_s = boundaries[idx + 1]
            self.tree.insert(
                "",
                "end",
                values=(
                    idx + 1,
                    f"{start_s:.2f}",
                    f"{end_s:.2f}",
                    f"{(end_s - start_s):.2f}",
                ),
            )

        self.status_var.set(f"Done. Created {len(boundaries) - 1} tracks.")
        messagebox.showinfo("Split complete", "Track export completed.")


def main() -> None:
    root = tk.Tk()
    app = DesktopApp(root)
    root.minsize(980, 680)
    root.mainloop()


if __name__ == "__main__":
    main()
