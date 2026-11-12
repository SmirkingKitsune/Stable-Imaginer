"""An engine process together with every process it starts: started and ended as one.

An engine is often more than one process. llama-server in router mode starts one per
model, and a wrapper (a ``.cmd`` script, a venv's python launcher) starts the real server
as its child. Ending only the top process leaves the rest running, holding the port, the
GPU memory and the log pipe. So each engine gets a container for its whole tree:

- Windows: a Job Object with KILL_ON_JOB_CLOSE. The engine starts suspended, joins the
  job, then resumes, so every process it starts is in the job too. Ending the job ends
  them all, and so does this server exiting, however it exits (the handle closes).
- POSIX: a new session, so the engine leads its own process group; signals go to the
  group.
"""

import contextlib
import os
import signal
import subprocess
import sys
import threading
from typing import Any

from imaginer.core.logging import get_logger

log = get_logger(__name__)


class ProcessTree:
    """A started engine and everything it starts. ``terminate``, ``kill`` and ``close``
    are safe from any thread and do nothing once the tree is closed."""

    def __init__(self, proc: subprocess.Popen[str], job: int | None = None) -> None:
        self.proc = proc
        self._job = job  # Windows Job Object handle, when the engine is in one
        self._lock = threading.Lock()
        self._closed = False

    @classmethod
    def spawn(cls, argv: list[str], **popen: Any) -> "ProcessTree":
        """Start ``argv`` with ``subprocess.Popen(argv, **popen)`` in a tree of its own."""
        # if/else statements, not early returns: mypy reports the other platform's code.
        if sys.platform == "win32":
            proc, job = _spawn_in_job(argv, popen)
        else:
            proc, job = subprocess.Popen(argv, start_new_session=True, **popen), None
        return cls(proc, job)

    def terminate(self) -> None:
        """Ask the whole tree to exit: SIGTERM on POSIX. Windows has no such request for a
        program without a console, so there it is the same as ``kill``."""
        with self._lock:
            if self._closed:
                return
            if sys.platform == "win32":
                self._kill_tree()
            else:
                self._signal_group(signal.SIGTERM)

    def kill(self) -> None:
        """End every process in the tree now."""
        with self._lock:
            if not self._closed:
                self._kill_tree()

    def close(self) -> None:
        """End whatever is left of the tree and let go of it. Called once the top process
        has exited: a wrapper's children or an engine's workers don't outlive it."""
        with self._lock:
            if self._closed:
                return
            self._kill_tree()
            self._closed = True
            if sys.platform == "win32" and self._job is not None:
                _kernel32.CloseHandle(self._job)
                self._job = None

    def _kill_tree(self) -> None:
        if sys.platform == "win32":
            if self._job is not None:
                _kernel32.TerminateJobObject(self._job, 1)  # 1, like Popen.terminate
                return
            # Not in a job (one couldn't be made): end the tree by parentage instead,
            # which misses processes whose parent has already exited.
            with contextlib.suppress(OSError):
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                    capture_output=True,
                    check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            with contextlib.suppress(OSError):
                self.proc.kill()
        else:
            self._signal_group(signal.SIGKILL)

    def _signal_group(self, sig: int) -> None:
        if sys.platform != "win32":
            # The group outlives its leader while any member is left, so this reaches a
            # wrapper's children after the wrapper itself has exited.
            with contextlib.suppress(OSError):  # the group is gone already
                os.killpg(self.proc.pid, sig)


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _CREATE_SUSPENDED = 0x00000004
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_SUSPEND_RESUME = 0x0800
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9  # JOBOBJECTINFOCLASS

    class _BasicLimits(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IoCounters(ctypes.Structure):
        _fields_ = tuple(
            (name, ctypes.c_ulonglong)
            for name in ("Read", "Write", "Other", "ReadBytes", "WriteBytes", "OtherBytes")
        )

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL
    # Undocumented but long-stable (psutil and Process Explorer resume processes with it).
    # The documented route, ResumeThread, needs the main thread's handle, which Popen closes.
    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)
    _ntdll.NtResumeProcess.restype = ctypes.c_long

    def _spawn_in_job(
        argv: list[str], popen: dict[str, Any]
    ) -> tuple[subprocess.Popen[str], int | None]:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            job = _new_job()
        except OSError as exc:
            log.warning("engine.no_job_object", error=str(exc))
            return subprocess.Popen(argv, creationflags=flags, **popen), None
        try:
            # Suspended until it is in the job, so nothing it starts can escape the job.
            proc = subprocess.Popen(argv, creationflags=flags | _CREATE_SUSPENDED, **popen)
            joined = _join_and_resume(proc, job)
        except BaseException:
            _kernel32.CloseHandle(job)  # ends the process too, if it joined
            raise
        if not joined:
            _kernel32.CloseHandle(job)
            return proc, None
        return proc, job

    def _new_job() -> int:
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.get_last_error()
            _kernel32.CloseHandle(job)
            raise ctypes.WinError(error)
        return int(job)

    def _join_and_resume(proc: subprocess.Popen[str], job: int) -> bool:
        """Put the suspended ``proc`` in ``job`` and let it run. False if it could not join
        (it runs anyway, outside the job). If it can't be resumed, it is ended and OSError
        raised."""
        access = _PROCESS_SET_QUOTA | _PROCESS_TERMINATE | _PROCESS_SUSPEND_RESUME
        handle = _kernel32.OpenProcess(access, False, proc.pid)
        try:
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            joined = bool(_kernel32.AssignProcessToJobObject(job, handle))
            if not joined:
                error = ctypes.WinError(ctypes.get_last_error())
                log.warning("engine.no_job_object", error=str(error))
            status = _ntdll.NtResumeProcess(handle)
            if status != 0:
                raise OSError(f"could not resume the engine process (NTSTATUS {status:#x})")
            return joined
        except BaseException:
            with contextlib.suppress(OSError):
                proc.kill()
            proc.wait()
            raise
        finally:
            if handle:
                _kernel32.CloseHandle(handle)
