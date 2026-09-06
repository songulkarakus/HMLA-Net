"""
SUPERVISOR: runs the pipeline and restarts it AUTOMATICALLY if it crashes or is interrupted.

Why is it needed?
  run_all.py / run_benchmark.py / run_ablation.py already support resume (checkpoints +
  DONE files) — but somebody has to RE-RUN the command. This script takes over that job:
  if the process crashes, or the power fails and the machine reboots, it continues
  where it left off by itself.

Usage:
    python auto_run.py                          # default: run_all.py full study
    python auto_run.py --cmd "run_external.py --experiments benchmark,proposed,ablation"
    python auto_run.py --max_retries 100 --delay 60
    python auto_run.py --status                 # show the status only, do not run
    python auto_run.py --reset                  # delete the completion marker (allow a fresh start)

To start automatically at Windows boot: auto_run.bat + Task Scheduler.

Features:
  - Lock file: two copies never run at once (Scheduler + manual start do not collide)
  - Completion marker: once the job is done it does not start again and again
  - Log: results/auto_run.log (every attempt time-stamped)
  - Exponential back-off: the wait grows after consecutive crashes (let GPU/disk recover)
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# keep non-ASCII characters from crashing the Windows console (cp1252/cp1254)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# This file lives under 'tools/'; the project root is one level up.
BASE = Path(__file__).resolve().parent.parent
STATE_DIR = BASE / "results"
LOCK = STATE_DIR / "auto_run.lock"
DONE = STATE_DIR / "auto_run.DONE"
LOG = STATE_DIR / "auto_run.log"

DEFAULT_CMD = ("run_all.py --seeds 42,1234,2024 --batch_size 32 "
               "--img_size 224 --num_workers 4")


def log(msg):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pid_alive(pid: int) -> bool:
    """Is the PID still running on Windows (via tasklist)?"""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True, timeout=15).stdout
        return str(pid) in out
    except Exception:
        return False           # if unsure, treat the lock as stale


def pipeline_already_running() -> str:
    """Is a manually started pipeline process running? (returns a description if so)

    The lock file only separates two supervisors; if the user also started run_all.py
    from PyCharm/a terminal, the supervisor could start a SECOND training writing to
    the same checkpoint files. This check prevents that.
    """
    scripts = ("run_all.py", "run_benchmark.py", "run_ablation.py",
               "run_optuna.py", "run_external.py")
    try:
        ps = ("Get-CimInstance Win32_Process | "
              "Where-Object { $_.Name -like 'python*' } | "
              "Select-Object -ExpandProperty CommandLine")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""
    me = str(os.getpid())
    for line in (out or "").splitlines():
        low = line.lower()
        if "auto_run.py" in low:
            continue                       # the supervisor itself
        if any(s in low for s in scripts) and me not in line:
            return line.strip()[:120]
    return ""


def acquire_lock() -> bool:
    """Returns False if another copy is running. Cleans up a stale lock."""
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text(encoding="utf-8").strip() or 0)
        except Exception:
            pid = 0
        if pid and pid != os.getpid() and _pid_alive(pid):
            log(f"Already running (PID {pid}) -> this copy exits.")
            return False
        log("Stale lock found (the previous run did not shut down cleanly) -> removed.")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock():
    try:
        if LOCK.exists() and LOCK.read_text(encoding="utf-8").strip() == str(os.getpid()):
            LOCK.unlink()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", default=DEFAULT_CMD,
                    help="python command to run (script + arguments)")
    ap.add_argument("--max_retries", type=int, default=200,
                    help="max attempts on consecutive failures")
    ap.add_argument("--delay", type=int, default=30,
                    help="seconds to wait before retrying (grows exponentially)")
    ap.add_argument("--max_delay", type=int, default=600, help="max wait (s)")
    ap.add_argument("--status", action="store_true", help="show the status and exit")
    ap.add_argument("--reset", action="store_true", help="delete the completion marker")
    args = ap.parse_args()

    if args.reset:
        for p in (DONE, LOCK):
            if p.exists():
                p.unlink()
                print(f"deleted: {p.name}")
        print("Reset; the job will be executed again on the next run.")
        return

    if args.status:
        print(f"completed : {'YES' if DONE.exists() else 'no'}")
        if DONE.exists():
            print(f"  {DONE.read_text(encoding='utf-8').strip()}")
        if LOCK.exists():
            pid = LOCK.read_text(encoding='utf-8').strip()
            print(f"lock      : PID {pid} ({'running' if _pid_alive(int(pid or 0)) else 'stale'})")
        else:
            print("lock      : none")
        print(f"log       : {LOG if LOG.exists() else '(not yet)'}")
        return

    if DONE.exists():
        log(f"The job is already complete ({DONE.read_text(encoding='utf-8').strip()}). "
            f"To run it again: python auto_run.py --reset")
        return

    busy = pipeline_already_running()
    if busy:
        log(f"A manually started run already exists -> the supervisor stays out.\n"
            f"  {busy}")
        return

    if not acquire_lock():
        return

    py = sys.executable
    cmd = [py] + args.cmd.split()
    log("=" * 70)
    log(f"SUPERVISOR started (PID {os.getpid()})")
    log(f"Command: {' '.join(cmd)}")
    log(f"Working directory: {BASE}")

    try:
        delay = args.delay
        for attempt in range(1, args.max_retries + 1):
            log(f"--- Attempt {attempt}/{args.max_retries} ---")
            t0 = time.time()
            try:
                rc = subprocess.run(cmd, cwd=str(BASE)).returncode
            except KeyboardInterrupt:
                log("Interrupted by the user (Ctrl+C). The supervisor exits; "
                    "restart it to resume.")
                return
            dur = time.time() - t0

            if rc == 0:
                DONE.write_text(
                    f"completed: {datetime.now():%Y-%m-%d %H:%M:%S} | "
                    f"command: {args.cmd}", encoding="utf-8")
                log(f"SUCCESS (duration {dur/60:.1f} min). Completion marker written.")
                return

            log(f"Exit code {rc} (duration {dur/60:.1f} min) -> resuming in {delay} s.")
            time.sleep(delay)
            # reset the wait if it ran for a long time before crashing, otherwise grow it
            delay = args.delay if dur > 600 else min(delay * 2, args.max_delay)

        log(f"WARNING: not completed after {args.max_retries} attempts. "
            f"Inspect the log: {LOG}")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
