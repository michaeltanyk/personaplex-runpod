#!/usr/bin/env python3
"""
WebSocket test client for PersonaPlex on RunPod Serverless.

Connects directly to the PersonaPlex WebSocket at wss://<ip>:8998/api/chat.
Sends audio from microphone, receives AI speech + text transcripts.

Usage:
    python test_client.py --url "wss://WORKER_IP:8998/api/chat"

Or with voice and text prompts:
    python test_client.py --url "wss://..." --voice NATF2 --text "You are a helpful assistant."
"""

import asyncio
import argparse
import json
import sys
import ssl
import wave
import io
import struct

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


async def send_audio_file(ws, wav_path: str):
    """Send a WAV file as binary audio frames."""
    with wave.open(wav_path, "rb") as wf:
        # Read WAV metadata
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()

        print(f"[send] {wav_path}: {framerate}Hz, {nchannels}ch, {sampwidth*8}bit, {nframes} frames")

        # Read in chunks and send
        chunk_size = 4096
        total_sent = 0
        while True:
            data = wf.readframes(chunk_size)
            if not data:
                break
            await ws.send(data)
            total_sent += len(data)
            await asyncio.sleep(0.02)  # Simulate real-time streaming

        print(f"[send] Done. Sent {total_sent} bytes.")


async def recv_audio(ws, output_path: str):
    """Receive audio from PersonaPlex and save to WAV file."""
    audio_frames = []
    text_transcript = []

    try:
        async for message in ws:
            if isinstance(message, bytes):
                audio_frames.append(message)
                print(f"[recv] Audio chunk: {len(message)} bytes")

            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                    if "text" in data:
                        text = data["text"]
                        text_transcript.append(text)
                        print(f"[recv] Text: {text}")
                except json.JSONDecodeError:
                    print(f"[recv] Raw text: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("[recv] Connection closed by server.")

    # Save audio to WAV
    if audio_frames and output_path:
        raw_audio = b"".join(audio_frames)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(24000)  # PersonaPlex uses 24kHz
            wf.writeframes(raw_audio)
        print(f"[recv] Saved {len(raw_audio)} bytes to {output_path}")

    # Print full transcript
    if text_transcript:
        print(f"\n[transcript] {' '.join(text_transcript)}")


async def main():
    parser = argparse.ArgumentParser(description="PersonaPlex WebSocket Test Client")
    parser.add_argument("--url", required=True, help="WebSocket URL (wss://ip:8998/api/chat)")
    parser.add_argument("--input", help="Input WAV file (16kHz mono recommended)")
    parser.add_argument("--output", default="personaplex_output.wav", help="Output WAV file")
    parser.add_argument("--voice", default="NATF2", help="Voice preset (NATF0-3, NATM0-3, VARF0-4, VARM0-4)")
    parser.add_argument("--text", default="You enjoy having a good conversation.",
                       help="Text role prompt")
    parser.add_argument("--insecure", action="store_true",
                       help="Accept self-signed SSL certs")
    args = parser.parse_args()

    ssl_context = None
    if args.insecure:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    print(f"[client] Connecting to {args.url}...")
    print(f"[client] Voice: {args.voice}, Text: {args.text}")

    async with websockets.connect(args.url, ssl=ssl_context) as ws:
        print("[client] Connected!")

        # Send voice prompt via JSON
        await ws.send(json.dumps({
            "voice": args.voice,
            "text": args.text,
        }))

        # Start receiving in background
        recv_task = asyncio.create_task(recv_audio(ws, args.output))

        # Send audio if provided
        if args.input:
            await send_audio_file(ws, args.input)
            print("[client] Audio sent. Waiting for response...")
        else:
            print("[client] No input audio. Listening only (press Ctrl+C to stop)...")
            try:
                await asyncio.sleep(300)  # 5 min timeout
            except KeyboardInterrupt:
                pass

        # Signal end of input
        await ws.send(json.dumps({"eos": True}))

        # Wait for response to finish
        await asyncio.sleep(3)
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    print("[client] Done.")


if __name__ == "__main__":
    asyncio.run(main())
