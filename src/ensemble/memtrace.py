"""Fine-grained resident-memory tracing for the long streaming stages.

The Africa scorecard was OOM-killed between two 120 s memory polls: the process went from
steady to dead inside one interval, so the poll never saw the peak and the stage that caused
it had to be inferred. This samples RSS on a daemon thread fast enough to catch an
allocation that lives for a second, attributes each sample to the labelled section that was
running, and keeps the whole trace so the shape of the peak (a slow climb vs a single
allocation) is visible afterwards.

Reading ``/proc/self/statm`` is two syscalls and a split; at 0.25 s it costs nothing next to
the work being measured, and it needs no third-party package.

``tracemalloc`` is offered alongside but off by default: it only sees Python-level
allocations (which is most of what matters here, since the suspects are all NumPy arrays
allocated from Python) and roughly doubles runtime.
"""

from __future__ import annotations

import csv
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PAGE = 4096


def rss_bytes() -> int:
    """Resident set size of this process, from /proc (Linux only)."""
    try:
        with open("/proc/self/statm", "rb") as f:
            return int(f.read().split()[1]) * PAGE
    except (OSError, IndexError, ValueError):
        return 0


class MemoryTrace:
    """Samples RSS on a background thread, labelled by the active section.

    Sections nest: ``with trace.section("aggregate"):`` inside it
    ``with trace.section("aggregate/block_member_stats")``. The label recorded with each
    sample is the innermost active one, so a peak lands on the narrowest section that was
    running rather than on the whole stage.
    """

    def __init__(self, out_path=None, interval: float = 0.25, tracemalloc: bool = False):
        self.out_path = Path(out_path) if out_path else None
        self.interval = float(interval)
        self.samples = []            # (t, rss, label)
        self.peaks = {}              # label -> peak rss while it was innermost
        self._stack = ["startup"]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._t0 = time.time()
        self._tracemalloc = bool(tracemalloc)
        self.baseline = rss_bytes()

    # -- lifecycle ---------------------------------------------------------------------
    def start(self):
        if self._tracemalloc:
            import tracemalloc
            tracemalloc.start(10)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1.0)
        if self._tracemalloc:
            import tracemalloc
            tracemalloc.stop()
        self.flush()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._record()

    def _record(self):
        r = rss_bytes()
        with self._lock:
            label = self._stack[-1]
            self.samples.append((time.time() - self._t0, r, label))
            # Every enclosing section owns the peak too, or a nested section would hide the
            # stage total from the stage's own row.
            for lab in self._stack:
                if r > self.peaks.get(lab, 0):
                    self.peaks[lab] = r

    # -- sections ----------------------------------------------------------------------
    @contextmanager
    def section(self, label):
        with self._lock:
            full = f"{self._stack[-1]}/{label}" if len(self._stack) > 1 else label
            self._stack.append(full)
        self._record()                       # bracket the section so a fast one is not missed
        t0 = time.time()
        try:
            yield self
        finally:
            self._record()
            with self._lock:
                self._stack.pop()
                peak = self.peaks.get(full, 0)
            print(f"    [mem] {full}: peak {peak / 1e9:.2f} GB "
                  f"(+{(peak - self.baseline) / 1e9:.2f} GB over baseline) "
                  f"in {time.time() - t0:.1f}s", flush=True)

    def snapshot(self, label, top=8):
        """Top ``tracemalloc`` allocation sites right now, if tracemalloc is enabled."""
        if not self._tracemalloc:
            return
        import tracemalloc
        cur, peak = tracemalloc.get_traced_memory()
        print(f"    [tracemalloc] {label}: current {cur / 1e9:.2f} GB, peak {peak / 1e9:.2f} GB")
        for i, st in enumerate(tracemalloc.take_snapshot().statistics("lineno")[:top], 1):
            f = st.traceback[0]
            print(f"      {i}. {st.size / 1e9:6.2f} GB  {f.filename}:{f.lineno}")
        tracemalloc.reset_peak()

    # -- output ------------------------------------------------------------------------
    @property
    def peak(self):
        return max((r for _, r, _ in self.samples), default=self.baseline)

    def flush(self):
        if not self.out_path:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        # Section labels carry commas (``scores[2005,B=1]``), so the field is quoted.
        with open(self.out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seconds", "rss_bytes", "section"])
            for t, r, lab in self.samples:
                w.writerow([f"{t:.3f}", r, lab])

    def report(self):
        lines = [f"peak RSS {self.peak / 1e9:.2f} GB (baseline {self.baseline / 1e9:.2f} GB)"]
        for lab, r in sorted(self.peaks.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {r / 1e9:8.2f} GB  {lab}")
        return "\n".join(lines)


class _NullTrace:
    """Same surface, no threads and no samples, so call sites need no conditionals."""

    peak = 0
    peaks = {}
    samples = []
    baseline = 0

    def start(self):
        return self

    def stop(self):
        return self

    @contextmanager
    def section(self, label):
        yield self

    def snapshot(self, label, top=8):
        pass

    def flush(self):
        pass

    def report(self):
        return "memory tracing disabled (--mem_trace)"


def make_trace(enabled, out_path=None, interval: float = 0.25, tracemalloc: bool = False):
    if not enabled:
        return _NullTrace()
    return MemoryTrace(out_path, interval=interval, tracemalloc=tracemalloc).start()
