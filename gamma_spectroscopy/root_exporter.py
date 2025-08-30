# uproot-only, compact implementation
from __future__ import annotations
import os, re
from pathlib import Path
import numpy as np
import awkward as ak
import uproot

# Match: DataR_CHA@Picoscope_run00007_3.root (A/B, 5 digits, part int)
_RE = re.compile(r"^DataR_CH[AB]@Picoscope_run(\d{5})_(\d+)\.root$")

def next_run_id(folder: str | os.PathLike) -> int:
    p = Path(folder); p.mkdir(parents=True, exist_ok=True)
    mx = -1
    for f in p.glob("DataR_CH*@Picoscope_run?????_*.root"):
        m = _RE.match(f.name)
        if m: mx = max(mx, int(m.group(1)))
    return mx + 1  # 0 if none

class RootWriter:
    """One ROOT file per channel; rolls parts when size >= max_mb."""
    def __init__(self, folder: str, ch_label: str, run_id: int, max_mb: int | None):
        assert ch_label in ("A", "B")
        self.dir = Path(folder); self.dir.mkdir(parents=True, exist_ok=True)
        self.ch = ch_label
        self.ch_idx = 0 if ch_label == "A" else 1
        self.run_id = int(run_id)
        self.part = 0
        self.max_bytes = int(max_mb*1024*1024) if max_mb else None
        self._buf = {k: [] for k in ("Channel","Timestamp","Board","Energy","Flags","Probe","Samples")}
        self._open()

    def add(self, samples_i16: np.ndarray, ts_ns: int, energy_u16: int,
            flags_u32: int = 0, probe_i32: int | None = None, board_u16: int = 0):
        if probe_i32 is None: probe_i32 = self.ch_idx
        self._buf["Channel"].append(np.uint16(self.ch_idx))
        self._buf["Timestamp"].append(np.uint64(ts_ns))
        self._buf["Board"].append(np.uint16(board_u16))
        self._buf["Energy"].append(np.uint16(energy_u16))
        self._buf["Flags"].append(np.uint32(flags_u32))
        self._buf["Probe"].append(np.int32(probe_i32))
        self._buf["Samples"].append(np.asarray(samples_i16, dtype=np.int16))
        if len(self._buf["Channel"]) >= 1000: self.flush()

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
        if self._tree is None:
            self._f["Data_R"] = payload
            self._tree = self._f["Data_R"]
        else:
            self._tree.extend(payload)
        for v in self._buf.values(): v.clear()
        self._roll()

    def close(self):
        self.flush()
        try: self._f.close()
        except: pass

    # ---- internals ----
    def _fname(self):  # DataR_CH{A|B}@Picoscope_run{NNNNN}_{part}.root
        return self.dir / f"DataR_CH{self.ch}@Picoscope_run{self.run_id:05d}_{self.part}.root"

    def _open(self):
        self.path = self._fname()
        self._f = uproot.recreate(str(self.path))
        self._tree = None

    def _roll(self):
        if not self.max_bytes: return
        try: size = os.path.getsize(self.path)
        except FileNotFoundError: return
        if size >= self.max_bytes:
            try: self._f.close()
            except: pass
            self.part += 1
            self._open()

