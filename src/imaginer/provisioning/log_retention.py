"""Bound provisioning logs by age, count, and size.

Adapted from Batch-Image-Interrogator; see LICENSE.Batch-Image-Interrogator.
Budgets apply per filename pattern so other logs cannot evict build diagnostics.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RetentionPolicy:
    """Size limits for one family of log files."""

    keep_files: int
    keep_total_bytes: int
    max_bytes: int
    tail_bytes: int

    def historical_files(self) -> int:
        """Slots for existing logs, reserving one for the run about to start."""
        return max(0, self.keep_files - 1)

    def historical_bytes(self) -> int:
        """Bytes for existing logs, reserving a full allowance for the new one.

        Without this reservation, `keep_files - 1` logs at `max_bytes` each plus
        a fresh `max_bytes` log would exceed the advertised total.
        """
        return max(0, self.keep_total_bytes - self.max_bytes)


def truncate_log(log_path: Path, tail_bytes: int) -> None:
    """Drop the head of an oversized log, keeping its recent tail.

    Safe to call on a file a live process still holds open, provided that
    process opened it in append mode: every subsequent write then lands at the
    new end of file rather than at a stale offset that would leave a sparse
    hole. Lines written during the swap can be lost, which is the deliberate
    trade for bounding a runaway log.
    """
    try:
        with log_path.open("rb") as handle:
            handle.seek(-tail_bytes, os.SEEK_END)
            tail = handle.read()
    except OSError:
        return

    # The tail almost certainly starts mid-line; drop the partial one.
    newline = tail.find(b"\n")
    if newline != -1:
        tail = tail[newline + 1 :]

    marker = (
        f"# [log truncated at {datetime.now().isoformat(timespec='seconds')}; "
        f"kept trailing {len(tail)} bytes]\n"
    ).encode()
    try:
        os.truncate(log_path, 0)
        with log_path.open("ab") as handle:
            handle.write(marker)
            handle.write(tail)
    except OSError:
        # Windows can refuse truncation while another process holds the file
        # open. Leaving the log oversized beats interrupting the writer.
        return


def prune_logs(log_dir: Path, pattern: str, policy: RetentionPolicy) -> None:
    """Bring one log family within its budget, newest first.

    Call before opening a new log, so the budget leaves room for the run about
    to start. Failures are ignored throughout: losing a diagnostic log is never
    a reason to refuse to do the actual work.
    """
    try:
        logs = sorted(
            (path for path in log_dir.glob(pattern) if path.is_file() and not path.is_symlink()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return

    keep_files = policy.historical_files()
    keep_bytes = policy.historical_bytes()
    kept_bytes = 0
    budget_exhausted = False
    for index, path in enumerate(logs):
        try:
            size = path.stat().st_size
        except OSError:
            continue

        # A previous unclean shutdown may have left an uncapped log. Keep its
        # useful tail before applying the historical-log budget.
        if size > policy.max_bytes:
            truncate_log(path, policy.tail_bytes)
            try:
                size = path.stat().st_size
            except OSError:
                continue

        if not budget_exhausted and index < keep_files and kept_bytes + size <= keep_bytes:
            kept_bytes += size
            continue
        budget_exhausted = True
        with contextlib.suppress(OSError):
            path.unlink()
