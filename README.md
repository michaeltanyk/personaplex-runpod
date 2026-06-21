# PersonaPlex on RunPod Serverless
# ==================================
# GPU-accelerated, pay-per-second, full-duplex voice AI
#
# Quick Start:
#   1. Build:  docker build -t yourname/personaplex-runpod .
#   2. Push:   docker push yourname/personaplex-runpod
#   3. Deploy: RunPod → Serverless → New Endpoint → A6000 (48GB)
#   4. Env vars: HF_TOKEN=your_huggingface_token
#   5. TCP port: Expose 8998
#   6. Wake:   curl -X POST https://api.runpod.ai/v2/ENDPOINT_ID/run -d '{"input":{}}'
#   7. Talk:   Connect WebSocket to returned URL
#
# Cost: ~$0.85/hr equivalent, billed per second while active
# Cold start: 2-5 min (model download + load into VRAM)
# Warm requests: <1s (model stays loaded)

## Requirements
- RunPod account with serverless access
- HuggingFace token with access to nvidia/personaplex-7b-v1
- Accept the model license: https://huggingface.co/nvidia/personaplex-7b-v1

## Architecture
```
┌──────────┐    REST (wake)    ┌──────────────────┐
│  Client  │ ────────────────▶ │  RunPod Serverless│
│          │                   │  ┌──────────────┐ │
│          │ ◀──── WS URL ──── │  │ rp_handler   │ │
│          │                   │  │   starts     │ │
│          │  WebSocket audio  │  │ PersonaPlex  │ │
│          │ ◀═══════════════▶ │  │  server      │ │
│          │  (full duplex!)   │  │  :8998       │ │
└──────────┘                   │  └──────────────┘ │
                               └──────────────────┘
```

## Testing locally after deployment
```bash
pip install websockets
python test_client.py --url "wss://WORKER_IP:8998/api/chat"
```
