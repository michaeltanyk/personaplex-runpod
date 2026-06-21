#!/usr/bin/env python3
"""Pre-download voice embeddings for PersonaPlex at build time."""

import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

REPO = "nvidia/personaplex-7b-v1"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[skip] No HF_TOKEN set — voices will download at runtime")
        return

    print(f"[download] Fetching voice embeddings from {REPO}...")
    try:
        # Download only the voice .pt files (small, ~few MB each)
        snapshot_download(
            REPO,
            token=token,
            allow_patterns=["*.pt", "voices/**"],
            ignore_patterns=["*.safetensors", "*.bin", "*.gguf"],
            local_dir="/app/hf_cache/personaplex-assets",
        )
        print("[download] Voice embeddings ready.")
    except Exception as e:
        print(f"[download] Non-fatal error: {e}")
        print("[download] Voices will be downloaded at runtime.")

if __name__ == "__main__":
    main()
