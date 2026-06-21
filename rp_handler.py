#!/usr/bin/env python3
"""
RunPod Serverless Handler for NVIDIA PersonaPlex
================================================
Architecture:
  1. Worker starts → loads PersonaPlex model into GPU VRAM (one-time, ~34GB)
  2. PersonaPlex WebSocket server runs on port 8998
  3. On each job: handler returns the WebSocket URL for direct client connection
  4. Client talks to PersonaPlex via WebSocket (wss:// or ws://)
  5. Worker stays warm between requests; scales to zero after idle timeout

WebSocket protocol (from manustik/personaplex-voice-bot):
  - Connect to wss://<host>:8998/api/chat
  - Send binary audio frames (Opus-encoded, 24kHz)
  - Receive binary audio frames (AI speech)
  - Receive text frames (transcripts)
"""

import os
import sys
import json
import time
import signal
import subprocess
import tempfile
import socket
import threading
import ssl
import http.server
import urllib.request

import runpod
from runpod.serverless.utils import rp_upload, rp_download


# ── Global state ────────────────────────────────────────────────────────────
_personaplex_process = None
_server_ready = threading.Event()
_public_ip = None


# ── PersonaPlex management ──────────────────────────────────────────────────

def get_public_ip():
    """Get the public IP of this worker."""
    try:
        return urllib.request.urlopen(
            "https://api.ipify.org", timeout=5
        ).read().decode().strip()
    except Exception:
        # Fallback: get from metadata service
        try:
            return urllib.request.urlopen(
                "http://169.254.169.254/latest/meta-data/public-ipv4",
                timeout=2,
            ).read().decode().strip()
        except Exception:
            return "127.0.0.1"


def start_personaplex(hf_token, voice="NATF2", cpu_offload=False):
    """Start the PersonaPlex server in background. Blocks until ready."""
    global _personaplex_process

    ssl_dir = tempfile.mkdtemp(prefix="personaplex-ssl-")

    cmd = [
        sys.executable, "-m", "moshi.server",
        "--ssl", ssl_dir,
        "--host", "0.0.0.0",
        "--port", "8998",
    ]

    if cpu_offload:
        cmd.append("--cpu-offload")

    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token

    print(f"[PersonaPlex] Starting server: {' '.join(cmd)}")
    _personaplex_process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for server to be ready by tailing output
    start_time = time.time()
    timeout = 600  # 10 min for model download + load

    for line in _personaplex_process.stdout:
        line = line.strip()
        print(f"[PersonaPlex] {line}")

        # Common ready signals
        if any(phrase in line.lower() for phrase in [
            "listening", "ready", "serving", "started",
            "running on", "uvicorn", "application startup",
        ]):
            _server_ready.set()
            print("[PersonaPlex] Server is ready!")
            break

        if time.time() - start_time > timeout:
            raise TimeoutError("PersonaPlex server failed to start within 10 minutes")

    # Continue reading stdout in background for logging
    def _log_stdout():
        for line in _personaplex_process.stdout:
            print(f"[PersonaPlex] {line.strip()}")

    threading.Thread(target=_log_stdout, daemon=True).start()


def stop_personaplex():
    """Gracefully stop the PersonaPlex server."""
    global _personaplex_process
    if _personaplex_process:
        print("[PersonaPlex] Stopping server...")
        _personaplex_process.send_signal(signal.SIGTERM)
        try:
            _personaplex_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[PersonaPlex] Force killing...")
            _personaplex_process.kill()
            _personaplex_process.wait()
        _personaplex_process = None
        print("[PersonaPlex] Stopped.")


# ── RunPod Handler ──────────────────────────────────────────────────────────

def handler(job):
    """
    RunPod handler — this runs for EACH job while the worker is alive.

    Returns connection info for PersonaPlex WebSocket.
    The model is already loaded and running.
    """
    global _public_ip

    job_input = job.get("input", {})

    # If user sends an action, handle it
    action = job_input.get("action", "connect")

    if action == "shutdown":
        stop_personaplex()
        return {"status": "shutdown", "message": "PersonaPlex server stopped"}

    if action == "status":
        return {
            "status": "ready" if _server_ready.is_set() else "loading",
            "websocket_url": f"wss://{_public_ip}:8998/api/chat",
            "model": "personaplex-7b-v1",
        }

    # Default: return WebSocket connection info
    # Wait for server to be ready (should already be ready for warm requests)
    if not _server_ready.wait(timeout=30):
        return {"status": "error", "message": "Server not ready after 30s wait"}

    return {
        "status": "ready",
        "websocket_url": f"wss://{_public_ip}:8998/api/chat",
        "model": "personaplex-7b-v1",
        "note": "Connect via WebSocket. Send binary for audio. Server accepts self-signed certs.",
    }


# ── Startup ─────────────────────────────────────────────────────────────────

def main():
    """Called once when the worker starts."""
    global _public_ip

    # Get config from env vars
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[ERROR] HF_TOKEN environment variable is required")
        print("  Set it in RunPod Serverless → Environment Variables")
        print("  Get your token at https://huggingface.co/settings/tokens")
        print("  Then accept the model license at https://huggingface.co/nvidia/personaplex-7b-v1")
        sys.exit(1)

    cpu_offload = os.environ.get("CPU_OFFLOAD", "0").lower() in ("1", "true", "yes")

    # Get public IP for WebSocket URL
    _public_ip = get_public_ip()
    print(f"[PersonaPlex] Worker public IP: {_public_ip}")

    # Start PersonaPlex (this blocks until the model is loaded)
    print("[PersonaPlex] Loading model (this takes 2-5 minutes for first load)...")
    start_personaplex(hf_token, cpu_offload=cpu_offload)

    # Start the RunPod worker loop
    print("[RunPod] Starting worker loop...")
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
