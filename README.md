# AmodalSeg — Interactive Amodal Segmentation App

Click any object mask → see its full shape revealed, even if occluded.

## Architecture

```
frontend/index.html     ← Single-file HTML/JS/CSS app (no framework)
backend/server.py       ← FastAPI REST API
pipeline/
  segmenter.py          ← SAM3/SAM2/SAM1 wrapper
  amodal_completer.py   ← Amodal completion pipeline
```

## How it works

1. **Upload image** → sent to `/segment`
2. **SAM auto-segments** all objects → returns mask list with bbox + area
3. **User clicks a mask** → mask_id stored in frontend state
4. **Click "Complete amodal shape"** → POST to `/amodal_complete`
5. Pipeline runs:
   - Occluder detection (InstaOrderNet or heuristic)
   - CLIP-based prompt selection
   - Stable Diffusion v2 iterative inpainting
   - Alpha blending → RGBA output
6. **RGBA PNG** displayed + available for download

## Setup

```bash
# 1. Create environment
conda create -n amodalseg python=3.11
conda activate amodalseg

# 2. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install SAM (choose one)
# SAM2 (recommended):
pip install git+https://github.com/facebookresearch/sam2.git
# Download checkpoint:
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# SAM1 (fallback):
pip install git+https://github.com/facebookresearch/segment-anything.git
# Download checkpoint:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# 5. (Optional) InstaOrderNet for precise occlusion ordering
pip install git+https://github.com/HnKnA/InstaOrderNet.git

# 6. Run backend
cd backend
python server.py
# → API running at http://localhost:8000

# 7. Open frontend
# Open frontend/index.html in browser (or serve with any static server)
python -m http.server 3000 --directory frontend
# → http://localhost:3000
```

## GPU requirements

| Model | VRAM needed |
|-------|-------------|
| SAM2 large | ~6 GB |
| SD v2 inpainting | ~8 GB |
| Both together | ~12 GB recommended |

For CPU-only: set `device="cpu"` in both classes — slower (~2-5 min/image).

## API reference

### `POST /segment`
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file | required | Input image (JPG/PNG) |
| `points_per_side` | int | 32 | SAM grid density |
| `pred_iou_thresh` | float | 0.88 | IOU confidence cutoff |
| `stability_score_thresh` | float | 0.95 | Mask stability cutoff |

Response:
```json
{
  "session_id": "uuid",
  "masks": [
    {
      "id": 0,
      "bbox": [x, y, w, h],
      "area": 12345,
      "predicted_iou": 0.92,
      "label": "object_0",
      "occluded": true,
      "segmentation": {"x": 10, "y": 20, "w": 100, "h": 80}
    }
  ],
  "image_size": [width, height]
}
```

### `POST /amodal_complete`
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `session_id` | str | required | From /segment response |
| `mask_id` | int | required | Which mask to complete |
| `text_query` | str | "" | Optional object description |
| `max_iter` | int | 3 | Max inpainting iterations |

Response: PNG binary (RGBA, transparent background)

## Extending the pipeline

To plug in the AP-ProtoSAM one-shot approach:
- Add a reference image upload to the frontend
- Call `pipeline/protosam_localizer.py` before segmentation
- Use prototype-guided similarity heatmap to improve occluder detection

See `pipeline/amodal_completer.py` → `_build_occluder_mask()` for the integration point.
# AP_ProtoSAM_Amodal
