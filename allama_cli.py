#!/usr/bin/env python3
"""
Allama CLI — allama serve / run / list / ps / stop / logs
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
ALLAMA_DIR  = Path(__file__).parent
ALLAMA_PORT = int(os.environ.get("ALLAMA_PORT", "9000"))
BASE_URL    = f"http://127.0.0.1:{ALLAMA_PORT}"
PID_FILE    = Path(os.environ.get("ALLAMA_PID_FILE", "/tmp/allama_watchdog.pid"))
LOG_FILE    = ALLAMA_DIR / "logs" / "allama.log"
PYTHON      = sys.executable
SERVER_SCRIPT = str(ALLAMA_DIR / "allama.py")

# ── Helpers ────────────────────────────────────────────────────────────────────
def _get(path: str, timeout: float = 3.0):
    """Simple HTTP GET, returns parsed JSON or None."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _is_running() -> bool:
    return _get("/health") is not None


def _wait_for_server(timeout: int = 30) -> bool:
    """Block until server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_running():
            return True
        time.sleep(0.5)
    return False


def _start_daemon() -> bool:
    """Start the watchdog daemon in background. Returns True if newly started."""
    if _is_running():
        return False

    # Spawn watchdog detached from terminal
    proc = subprocess.Popen(
        [PYTHON, __file__, "__watchdog__"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    return True


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


# ── Watchdog (internal, not user-facing) ───────────────────────────────────────
def _run_watchdog(verbose: bool):
    """
    Loop forever restarting allama.py if it dies.
    Called internally when this script is run as __watchdog__.
    """
    PID_FILE.write_text(str(os.getpid()))
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _sigterm(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm)

    restart_count = 0
    while True:
        if verbose:
            proc = subprocess.run([PYTHON, SERVER_SCRIPT])
        else:
            with open(LOG_FILE, "a") as log:
                proc = subprocess.run(
                    [PYTHON, SERVER_SCRIPT],
                    stdout=log,
                    stderr=log,
                )

        code = proc.returncode

        # Clean shutdown (SIGINT / ctrl+c forwarded) — don't restart
        if code in (0, -signal.SIGINT, 130):
            break

        restart_count += 1
        msg = f"[allama] process exited (code {code}), restarting in 3s... (#{restart_count})"
        if verbose:
            print(msg, flush=True)
        else:
            with open(LOG_FILE, "a") as log:
                log.write(msg + "\n")
        time.sleep(3)

    PID_FILE.unlink(missing_ok=True)


# ── Commands ───────────────────────────────────────────────────────────────────
def cmd_serve(args):
    """Start the Allama daemon."""
    if args.verbose:
        # Foreground: run watchdog directly (shows rich banner + logs + auto-restart)
        print("Starting Allama (verbose mode — Ctrl+C to stop)...")
        _run_watchdog(verbose=True)
    else:
        if _is_running():
            print("Allama is already running.")
            return
        _start_daemon()
        print("Starting Allama...", end="", flush=True)
        if _wait_for_server(30):
            print(" ready.")
        else:
            print(" timed out. Check logs: allama logs")


def cmd_stop(args):
    """Stop the Allama daemon."""
    pid = _read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Stopped (PID {pid}).")
            PID_FILE.unlink(missing_ok=True)
            return
        except ProcessLookupError:
            PID_FILE.unlink(missing_ok=True)

    if not _is_running():
        print("Allama is not running.")
        return

    # Fallback: ask the server to shut itself down isn't standard, just warn
    print("Could not find watchdog PID. Kill allama.py manually if needed.")


def cmd_status(args):
    """Show server status and loaded models."""
    health = _get("/health")
    if not health:
        print("● Allama is not running")
        return

    active = health.get("active_servers", 0)
    print(f"● Allama is running  (port {ALLAMA_PORT})")
    print(f"  Loaded models: {active}")

    ps_data = _get("/v1/models")
    if ps_data:
        models = [m["id"] for m in ps_data.get("data", [])]
        if models:
            print(f"  Available ({len(models)}):")
            for m in models:
                print(f"    · {m}")


def cmd_list(args):
    """List available logical models."""
    data = _get("/v1/models")
    if data is None:
        print("Allama is not running. Start with: allama serve")
        return
    models = [m["id"] for m in data.get("data", [])]
    if not models:
        print("No models configured.")
        return
    for m in sorted(models):
        print(m)


def cmd_ps(args):
    """Show currently loaded (running) models."""
    health = _get("/health")
    if not health:
        print("Allama is not running.")
        return
    active = health.get("active_servers", 0)
    if active == 0:
        print("No models loaded.")
    else:
        print(f"{active} model(s) loaded.")


def cmd_logs(args):
    """Tail the Allama log file."""
    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        return
    try:
        if args.follow:
            subprocess.run(["tail", "-f", str(LOG_FILE)])
        else:
            lines = args.lines
            subprocess.run(["tail", f"-{lines}", str(LOG_FILE)])
    except KeyboardInterrupt:
        pass


def cmd_run(args):
    """Load a model and open an interactive chat session."""
    model = args.model

    # Ensure server is up
    if not _is_running():
        print(f"Starting Allama...", end="", flush=True)
        _start_daemon()
        if not _wait_for_server(30):
            print(" failed to start. Check logs: allama logs")
            sys.exit(1)
        print(" ready.")

    # Verify model exists
    data = _get("/v1/models")
    available = [m["id"] for m in (data or {}).get("data", [])]
    if model not in available:
        print(f"Model '{model}' not found.")
        if available:
            print("Available models:")
            for m in sorted(available):
                print(f"  {m}")
        sys.exit(1)

    _repl(model)


def _repl(model: str):
    """Interactive chat REPL for a model."""
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Run: pip install httpx")
        sys.exit(1)

    history = []
    print(f"\n{'─'*50}")
    print(f"  Model : {model}")
    print(f"  Port  : {ALLAMA_PORT}")
    print(f"  /bye or Ctrl+C to exit")
    print(f"{'─'*50}\n")
    print("Loading model on first message...\n")

    try:
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/bye", "/exit", "/quit"):
                print("Bye!")
                break
            if user_input == "/clear":
                history.clear()
                print("History cleared.")
                continue

            history.append({"role": "user", "content": user_input})

            payload = {
                "model": model,
                "messages": history,
                "stream": True,
            }

            full_response = ""
            print()
            try:
                with httpx.Client(timeout=300.0) as client:
                    with client.stream(
                        "POST",
                        f"{BASE_URL}/v1/chat/completions",
                        json=payload,
                        headers={"Authorization": "Bearer dummy"},
                    ) as resp:
                        if resp.status_code != 200:
                            print(f"[error {resp.status_code}]")
                            history.pop()
                            continue
                        for line in resp.iter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    chunk = json.loads(line[6:])
                                    delta = chunk["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        print(delta, end="", flush=True)
                                        full_response += delta
                                except (json.JSONDecodeError, KeyError):
                                    pass
            except KeyboardInterrupt:
                print("\n[interrupted]")
                history.pop()
                print()
                continue
            except httpx.ConnectError:
                print("[connection error — is Allama running?]")
                break

            print("\n")
            if full_response:
                history.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\nBye!")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    # Internal: running as watchdog daemon
    if len(sys.argv) > 1 and sys.argv[1] == "__watchdog__":
        verbose = "--verbose" in sys.argv
        _run_watchdog(verbose=verbose)
        return

    parser = argparse.ArgumentParser(
        prog="allama",
        description="Allama — local LLM manager",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # serve
    p_serve = sub.add_parser("serve", help="Start the Allama server")
    p_serve.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run in foreground with live logs and rich display",
    )
    p_serve.set_defaults(func=cmd_serve)

    # stop
    p_stop = sub.add_parser("stop", help="Stop the Allama server")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show server status")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = sub.add_parser("list", help="List available models")
    p_list.set_defaults(func=cmd_list)

    # ps
    p_ps = sub.add_parser("ps", help="Show loaded models")
    p_ps.set_defaults(func=cmd_ps)

    # logs
    p_logs = sub.add_parser("logs", help="Show Allama logs")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    p_logs.add_argument("-n", "--lines", type=int, default=50, metavar="N", help="Lines to show (default: 50)")
    p_logs.set_defaults(func=cmd_logs)

    # run
    p_run = sub.add_parser("run", help="Chat with a model interactively")
    p_run.add_argument("model", help="Logical model name (e.g. Qwen3.5:27b-Instruct)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
