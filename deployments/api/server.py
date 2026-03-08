"""
FastAPI inference server for VL-JEPA.

Provides REST API endpoints for:
- Image/video classification
- Text-to-video retrieval
- Visual question answering
- Embedding extraction
"""

from __future__ import annotations

import io
import time
from typing import Any

import torch
from PIL import Image

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError("Install deployment deps: pip install -e '.[deployment]'")

app = FastAPI(
    title="VL-JEPA Inference API",
    description="Vision-Language Joint Embedding Predictive Architecture",
    version="0.1.0",
)

# Global model (loaded on startup)
model = None
device = None


@app.on_event("startup")
async def load_model():
    """Load VL-JEPA model on server startup."""
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: Load from checkpoint
    # model = build_vljepa(config).to(device)
    # model.eval()
    print(f"VL-JEPA server ready on {device}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/embed/image")
async def embed_image(
    image: UploadFile = File(...),
    query: str = Form(default="What is in this image?"),
) -> dict[str, Any]:
    """Extract embedding for an image + query pair."""
    start = time.time()

    # Read and preprocess image
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # TODO: Preprocess and run inference
    # embedding = model.forward_embed(img_tensor, query_ids)

    elapsed = time.time() - start
    return {
        "embedding_dim": 1536,
        "inference_time_ms": round(elapsed * 1000, 2),
        "message": "Model not yet loaded. Deploy with checkpoint.",
    }


@app.post("/classify")
async def classify(
    image: UploadFile = File(...),
    labels: str = Form(...),  # Comma-separated class labels
) -> dict[str, Any]:
    """Zero-shot image classification."""
    label_list = [l.strip() for l in labels.split(",")]

    return {
        "labels": label_list,
        "message": "Model not yet loaded. Deploy with checkpoint.",
    }


@app.post("/vqa")
async def visual_qa(
    image: UploadFile = File(...),
    question: str = Form(...),
) -> dict[str, Any]:
    """Visual question answering."""
    return {
        "question": question,
        "message": "Model not yet loaded. Deploy with checkpoint.",
    }
