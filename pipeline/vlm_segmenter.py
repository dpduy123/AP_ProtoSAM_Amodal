"""
pipeline/vlm_segmenter.py — Stage 1: VLM-based Target Segmentation

Paper §3.1: Given text query Q, use LISA (VLM) to generate visible mask M_visible.

Architecture decision (per user): Run LISA first, save all masks, then unload LISA
before running the rest of the pipeline. LISA is accessed via Gradio API to avoid
dependency conflicts (LISA uses its own transformers/torch versions).

Reference: LISA — Reasoning Segmentation via Large Language Model (CVPR 2024)
           https://github.com/dvlab-research/LISA
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image


class VLMSegmenter:
    """
    Wraps LISA VLM for text-query-driven visible mask generation.

    LISA runs as a separate Gradio server. This class connects via gradio_client.
    The server must be started separately before calling this class:
        cd LISA && python app.py

    Usage:
        segmenter = VLMSegmenter(server_url="http://127.0.0.1:7860/")
        mask = segmenter.segment(image_path, "the polar bear in this image")
    """

    def __init__(self, server_url: str = "http://127.0.0.1:7860/", cache_dir: str = "./output/lisa_masks/"):
        self.server_url = server_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self):
        """Lazy-load gradio client to avoid import errors when LISA server is not running."""
        if self._client is None:
            from gradio_client import Client
            print(f"[VLMSegmenter] Connecting to LISA server at {self.server_url}...")
            self._client = Client(self.server_url)
            print("[VLMSegmenter] Connected.")
        return self._client

    def segment(self, image_path: str, text_query: str, use_cache: bool = True) -> np.ndarray:
        """
        Get visible mask for the target object specified by text query.

        Args:
            image_path: Absolute path to the input image.
            text_query: Natural language query (e.g., "polar bear", "the animal in this image").
            use_cache: If True, check cache before querying LISA.

        Returns:
            H×W bool numpy array — visible mask of the target object.
        """
        img_basename = Path(image_path).stem
        cache_path = self.cache_dir / f"{img_basename}.pkl"

        # Check cache first
        if use_cache and cache_path.exists():
            try:
                print(f"[VLMSegmenter] Loading cached mask for '{img_basename}'")
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if text_query in cached:
                    return cached[text_query]
            except Exception as e:
                print(f"[VLMSegmenter] Cache load failed ({e}), re-querying LISA...")

        # Query LISA server
        print(f"[VLMSegmenter] Querying LISA: '{text_query}' on '{img_basename}'...")
        client = self._get_client()
        result = client.predict(
            text_query,         # Text instruction
            image_path,         # Image path
            api_name="/predict"
        )

        # Parse result — LISA returns (output_image_path, mask_json_path)
        output_image_path = result[0]
        mask_json_path = result[1]

        with open(mask_json_path, "r") as f:
            mask_data = json.load(f)

        mask = np.array(mask_data["data"], dtype=bool)
        print(f"[VLMSegmenter] Mask obtained: {mask.shape}, area={mask.sum()} px")

        # Save to cache
        if use_cache:
            cached = {}
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached = pickle.load(f)
                except Exception:
                    cached = {}
            cached[text_query] = mask
            with open(cache_path, "wb") as f:
                pickle.dump(cached, f)

        return mask

    def segment_batch(self, image_paths: list[str], text_queries: list[str],
                      output_dir: Optional[str] = None) -> dict[str, np.ndarray]:
        """
        Process multiple images, saving all masks. Designed to be run BEFORE
        the main pipeline so LISA can be unloaded from VRAM.

        Args:
            image_paths: List of absolute paths to images.
            text_queries: Corresponding text queries (same length as image_paths).
            output_dir: Override output directory.

        Returns:
            Dict mapping image_basename → visible_mask
        """
        assert len(image_paths) == len(text_queries), "image_paths and text_queries must have same length"

        results = {}
        save_dir = Path(output_dir) if output_dir else self.cache_dir

        for img_path, query in zip(image_paths, text_queries):
            basename = Path(img_path).stem
            try:
                mask = self.segment(img_path, query)
                results[basename] = mask

                # Also save visible mask as PNG for visual inspection
                vis_path = save_dir / f"{basename}_visible_mask.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(vis_path)
                print(f"[VLMSegmenter] ✅ {basename}: mask saved")

            except Exception as e:
                print(f"[VLMSegmenter] ❌ {basename}: {e}")
                results[basename] = None

        return results

    def load_cached_mask(self, image_path: str, text_query: str) -> Optional[np.ndarray]:
        """Load a previously cached mask without connecting to LISA server."""
        cache_path = self.cache_dir / f"{Path(image_path).stem}.pkl"
        if not cache_path.exists():
            return None
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        return cached.get(text_query)
