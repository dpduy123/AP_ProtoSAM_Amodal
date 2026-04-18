"""
pipeline/config.py — Global configuration for the CVPR 2025 pipeline.

All model paths, hyperparameters, and runtime settings in one place.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration matching the paper's default settings."""

    # ── Device ──
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # ── LISA VLM Server ──
    lisa_server_url: str = "http://127.0.0.1:7860/"
    lisa_output_path: str = "./output/lisa_masks/"

    # ── Grounding DINO ──
    gdino_config: str = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    gdino_ckpt: str = "Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
    gdino_box_thresh: float = 0.35
    gdino_text_thresh: float = 0.35

    # ── SAM ──
    sam_ckpt: str = "Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

    # ── RAM++ ──
    ram_ckpt: str = "./recognize-anything/ram_plus_swin_large_14m.pth"
    ram_image_size: int = 384

    # ── InstaOrderNet ──
    instaorder_ckpt: str = "InstaOrder/InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar"
    instaorder_input_size: int = 384

    # ── Stable Diffusion v2 Inpainting ──
    sd_model_id: str = "stabilityai/stable-diffusion-2-inpainting"
    sd_target_size: int = 512

    # ── CLIP ──
    clip_model_name: str = "ViT-B/32"

    # ── Pipeline Hyperparameters ──
    max_iter: int = 3                   # Maximum inpainting iterations (paper default)
    epsilon: float = 0.01               # Occluder mask stability threshold (Eq. 7)
    canvas_multiplier: int = 6          # Canvas expansion factor
    canvas_fill_value: int = 255        # White background for canvas
    crop_buffer: int = 60               # Pixels of buffer around target when cropping
    pad_pixels: int = 150               # Pixels of padding for boundary-touching objects
    boundary_gap_pixels: int = 10       # How close to edge counts as "touching boundary"

    # ── Validation Thresholds ──
    query_pred_score_thresh: float = 0.3    # Min prediction score for valid query
    query_mask_size_thresh: float = 0.01    # Min mask area ratio

    # ── Clean Background ──
    clean_bkgd_image: str = "images/gray_wallpaper.jpeg"

    # ── Output ──
    output_dir: str = "./output"
    save_intermediate: bool = True

    # ── Hugging Face Cache ──
    hf_cache_dir: Optional[str] = None

    def __post_init__(self):
        """Set HuggingFace cache dirs if specified."""
        if self.hf_cache_dir:
            os.environ["HF_DATASETS_CACHE"] = self.hf_cache_dir
            os.environ["HF_HOME"] = self.hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = self.hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.hf_cache_dir

    def get_gpu_info(self) -> dict:
        """Return GPU information for adaptive settings."""
        if not torch.cuda.is_available():
            return {"name": "CPU", "vram_gb": 0, "device": "cpu"}
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "device": "cuda",
        }
