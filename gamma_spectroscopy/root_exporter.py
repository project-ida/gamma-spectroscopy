# uproot-only, compact implementation
from __future__ import annotations
import os, re
from pathlib import Path
import numpy as np
import awkward as ak
import uproot
import threading, queue  # ← async writer

# Match: DataR_CHA@Picoscope_run00007_3.root (A/B, 5 digits, part int)
_RE = re.compile(r"^DataR_CH[AB]@Picoscope_run(\d{5})_(\d+)\.root$")

def next_run_id(folder: str | os.PathLike) -> int:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)

    mx = -1

    # New layout: per-run subfolders named "runNNNNN"
    for d in p.iterdir():
        if d.is_dir():
            name = d.name
            if name.startswith("run") and len(name) == 8 and name[3:].isdigit():
                mx = max(mx, int(name[3:]))

    # Legacy layout: files directly in <folder>
    for f in p.glob("DataR_CH*@Picoscope_run?????_*.root"):
        m = _RE.match(f.name)
        if m:
            mx = max(mx, int(m.group(1)))

    return mx + 1  # 0 if none

class RootWriter:
    """One ROOT file per channel; rolls parts when size >= max_mb.

    Parameters
    ----------
    folder : str
        Output directory (per run, typically .../runNNNNN/RAW).
    ch_label : {"A","B"}
        Channel label written in the filename.
    run_id : int
        Run number used in the filename.
    max_mb : int | None
        If set, roll to a new file when size reaches this many MB.
    flush_every : int
        Flush buffer to disk every N events (default 4000).
    compression : any
        Passed through to uproot.recreate(..., compression=...).
        For speed you may pass uproot.LZ4(4) if lz4 is available.
    """
    def __init__(self, folder: str, ch_label: str, run_id: int, max_mb: int | None,
                 flush_every: int = 4000, compression=None):
        assert ch_label in ("A", "B")
        self.dir = Path(folder) 
        self.ch = ch_label
        self.ch_idx = 0 if ch_label == "A" else 1
        self.run_id = int(run_id)
        self.part = 0
        self.max_bytes = int(max_mb*1024*1024) if max_mb else None
        self.flush_every = int(flush_every)
        self._compression = compression
        self._buf = {k: [] for k in ("Channel","Timestamp","Board","Energy","Flags","Probe","Samples")}
        self._f = None
        self._tree = None
        self.path: Path | None = None  # current on-disk path (tmp while writing)
        self._open()

        # --- async writer thread: the file handle is only used by this thread ---
        self._q = queue.Queue(maxsize=16)  # holds ("WRITE", payload) or ("FINALIZE", None)
        self._writer = threading.Thread(target=self._writer_loop, name=f"RootWriter-{self.ch}", daemon=True)
        self._writer.start()

    def add(self, samples_i16: np.ndarray, ts_ns: int, energy_u16: int,
            flags_u32: int = 0, probe_i32: int | None = None, board_u16: int = 0):
        """Backward-compatible single-event add (timestamps stored in picoseconds)."""
        # Delegate to batched path for consistency and less code duplication
        self.add_many(
            samples_i16=np.asarray(samples_i16, dtype=np.int16).reshape(1, -1),
            ts_ns=np.asarray([ts_ns], dtype=np.uint64),
            energy_u16=np.asarray([energy_u16], dtype=np.uint16),
            flags_u32=np.asarray([flags_u32], dtype=np.uint32),
            probe_i32=np.asarray([self.ch_idx if probe_i32 is None else probe_i32], dtype=np.int32),
            board_u16=np.asarray([board_u16], dtype=np.uint16),
        )

    def add_many(self,
                 samples_i16: np.ndarray,
                 ts_ns: np.ndarray,
                 energy_u16: np.ndarray,
                 *,
                 flags_u32: np.ndarray | None = None,
                 probe_i32: np.ndarray | None = None,
                 board_u16: np.ndarray | None = None):
        """Append a block of N events in one go (timestamps in ns → stored as ps)."""
        samples_i16 = np.asarray(samples_i16, dtype=np.int16)
        if samples_i16.ndim == 1:
            samples_i16 = samples_i16.reshape(1, -1)
        N = int(samples_i16.shape[0])

        # Convert and shape metadata
        ts_ns = np.asarray(ts_ns, dtype=np.uint64).reshape(N)
        ts_ps = (ts_ns.astype(np.uint64)) * np.uint64(1000)  # store in picoseconds

        energy_u16 = np.asarray(energy_u16, dtype=np.uint16).reshape(N)
        flags_u32 = np.zeros(N, dtype=np.uint32) if flags_u32 is None else np.asarray(flags_u32, dtype=np.uint32).reshape(N)
        board_u16 = np.zeros(N, dtype=np.uint16) if board_u16 is None else np.asarray(board_u16, dtype=np.uint16).reshape(N)
        probe_i32 = (np.full(N, self.ch_idx, dtype=np.int32) if probe_i32 is None
                     else np.asarray(probe_i32, dtype=np.int32).reshape(N))

        # Extend buffers
        if N:
            self._buf["Channel"].extend([np.uint16(self.ch_idx)] * N)
            self._buf["Timestamp"].extend(ts_ps)
            self._buf["Board"].extend(board_u16)
            self._buf["Energy"].extend(energy_u16)
            self._buf["Flags"].extend(flags_u32)
            self._buf["Probe"].extend(probe_i32)
            # ensure each waveform is an owned, contiguous array
            self._buf["Samples"].extend([np.ascontiguousarray(row).copy() for row in samples_i16])

        if len(self._buf["Channel"]) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buf["Channel"]: return
        payload = {
            "Channel":   np.asarray(self._buf["Channel"],  dtype=np.uint16),
            "Timestamp": np.asarray(self._buf["Timestamp"], dtype=np.uint64),
            "Board":     np.asarray(self._buf["Board"],     dtype=np.uint16),
            "Energy":    np.asarray(self._buf["Energy"],    dtype=np.uint16),
            "Flags":     np.asarray(self._buf["Flags"],     dtype=np.uint32),
            "Probe":     np.asarray(self._buf["Probe"],     dtype=np.int32),
            "Samples":   ak.Array(self._buf["Samples"]),  # jagged int16
        }
        # Hand off the disk write to the background thread
        self._q.put(("WRITE", payload))
        for v in self._buf.values(): v.clear()

    def close(self):
        """Flush outstanding buffers, finalize .tmp → .root, and stop the worker."""
        try:
            if self._buf["Channel"]:
                self.flush()  # enqueues last payload
            self._q.put(("FINALIZE", None))
            self._q.join()  # wait until writer processed everything
            if self._writer.is_alive():
                self._writer.join(timeout=2.0)
        except Exception:
            pass

    # ---- internals ----
    def _fname_final(self) -> Path:  # DataR_CH{A|B}@Picoscope_run{NNNNN}_{part}.root
        return self.dir / f"DataR_CH{self.ch}@Picoscope_run{self.run_id:05d}_{self.part}.root"

    def _fname_tmp(self) -> Path:    # temp while writing, visible as .tmp
        return self.dir / f"DataR_CH{self.ch}@Picoscope_run{self.run_id:05d}_{self.part}.tmp"

    def _open(self):
        """Open a new .tmp file for the current part."""
        self.path = self._fname_tmp()
        self._f = uproot.recreate(str(self.path), compression=self._compression)
        self._tree = None

    def _finalize_file(self):
        """Close the current file (if open) and rename .tmp -> .root."""
        try:
            if self._f is not None:
                try:
                    self._f.close()
                finally:
                    self._f = None
                    self._tree = None
        finally:
            try:
                if self.path is not None:
                    p = Path(self.path)
                    if p.suffix == ".tmp" and p.exists():
                        final = self._fname_final()
                        # replace() will overwrite if a stale .root exists
                        p.replace(final)
                        self.path = final
            except Exception:
                # non-fatal: if rename fails, we leave the .tmp in place
                pass

    def _writer_loop(self):
        """Owns all I/O to the ROOT file: writes, rolling, and finalization."""
        while True:
            cmd, payload = self._q.get()
            try:
                if cmd == "WRITE":
                    if self._tree is None:
                        self._f["Data_R"] = payload
                        self._tree = self._f["Data_R"]
                    else:
                        self._tree.extend(payload)
                    # rolling is done on the writer thread
                    self._roll()
                elif cmd == "FINALIZE":
                    self._finalize_file()
                    return
            finally:
                self._q.task_done()

    def _roll(self):
        """If size threshold reached, finalize current .tmp and open next .tmp part."""
        if not self.max_bytes:
            return
        try:
            sz = os.path.getsize(self.path) if self.path else 0
        except FileNotFoundError:
            return
        if sz >= self.max_bytes:
            # finalize current file -> .root, then open next .tmp
            self._finalize_file()
            self.part += 1
            self._open()
