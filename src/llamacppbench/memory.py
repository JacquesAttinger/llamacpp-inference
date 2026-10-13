# Last edited: 2026-10-13 10:42
"""Peak memory helpers: a background thread polls the process's memory and keeps the maximum."""

import threading

import psutil

GB = 1e9
POLL_SECONDS = 3 / 1000  # Check every three milliseconds

# State shared by the functions below. Functions that assign to these need `global`.
_peak_bytes = 0  # running maximum of the process's resident memory
_stop = threading.Event()  # set to tell the polling thread to quit
_thread: threading.Thread | None = None
_process = psutil.Process()


def read_peak() -> None:
    """Thread body: update the running maximum until _stop is set."""
    global _peak_bytes
    # _stop.wait() sleeps for POLL_SECONDS, and wakes up early (returning True) when _stop is set.
    while not _stop.wait(POLL_SECONDS):
        _peak_bytes = max(_peak_bytes, _process.memory_info().rss)


def _stop_thread() -> None:
    global _thread
    if _thread is not None:
        _stop.set()
        _thread.join()
        _thread = None


def reset_peak() -> None:
    """Reset the peak-memory counter before a measured run."""
    global _peak_bytes, _thread
    _stop_thread()  # never leave an old thread running
    _stop.clear()
    _peak_bytes = _process.memory_info().rss  # one reading now, in case the run is shorter than a poll
    _thread = threading.Thread(target=read_peak, daemon=True)  # daemon: dies with the program
    _thread.start()


def peak_gb() -> float:
    """Peak memory in GB since the last reset (a close lower bound: spikes between polls are missed)."""
    global _peak_bytes
    _stop_thread()
    _peak_bytes = max(_peak_bytes, _process.memory_info().rss)  # one last reading
    return _peak_bytes / GB
