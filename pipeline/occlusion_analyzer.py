"""
pipeline/occlusion_analyzer.py — Stage 3: Occlusion Order Analysis

Paper §3.2: Use InstaOrderNet to determine pairwise occlusion relationships
between the target object and all other segments (objects + background).

Key equations:
  Eq. 2: M_occ = ∪(S_i where occ_i=1) ∪ ∪(B_j where occ_j=1)
  Eq. 3: M_occ ← M_occ ∪ d(M_visible) ∩ edge_e  (boundary expansion)

Reference:
  InstaOrder: Instance-wise Occlusion and Depth Orders in Natural Scenes (CVPR 2022)
  https://github.com/POSTECH-CVLab/InstaOrder
"""

import sys
import os
import gc
import numpy as np
import cv2
import torch
from typing import Optional

from pipeline.config import PipelineConfig


class OcclusionAnalyzer:
    """
    Determines which scene segments occlude the target object using InstaOrderNet.
    Produces an aggregated occluder mask for the inpainting stage.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self._model = None
        self._loaded = False

    def load_model(self):
        """Load InstaOrderNet model."""
        if self._loaded:
            return

        sys.path.append("InstaOrder")
        import models

        print("[OcclusionAnalyzer] Loading InstaOrderNet...")
        model_params = {
            "algo": "InstaOrderNet_od",
            "total_iter": 60000,
            "lr_steps": [32000, 48000],
            "lr_mults": [0.1, 0.1],
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "optim": "SGD",
            "warmup_lr": [],
            "warmup_steps": [],
            "use_rgb": True,
            "backbone_arch": "resnet50_cls",
            "backbone_param": {"in_channels": 5, "num_classes": [2, 3]},
            "overlap_weight": 0.1,
            "distinct_weight": 0.9,
        }
        self._model = models.__dict__["InstaOrderNet_od"](model_params)
        self._model.load_state(self.config.instaorder_ckpt)
        self._model.switch_to("eval")
        self._loaded = True
        print("[OcclusionAnalyzer] InstaOrderNet loaded.")

    def unload_model(self):
        """Free GPU memory."""
        del self._model
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()

    # ── Main entry point ─────────────────────────────────────────────────

    def analyze(self, image: np.ndarray, masks: np.ndarray,
                target_idx: int) -> np.ndarray:
        """
        Determine which masks occlude the target and produce aggregated occluder mask.

        Paper Eq. 2: M_occ = ∪(segments where InstaOrder says they occlude target)

        Args:
            image: H×W×3 uint8 RGB
            masks: (N, H, W) array of all masks (objects + bg + target)
            target_idx: Index of the target mask in `masks`

        Returns:
            H×W uint8 occluder mask (1 = occluder, 0 = not)
        """
        if not self._loaded:
            self.load_model()

        sys.path.append("InstaOrder")
        import inference as infer

        # Prepare modal masks array
        modal = np.zeros((len(masks), masks[0].shape[0], masks[0].shape[1]))
        for i, mask in enumerate(masks):
            modal[i] = mask

        # Compute expanded bounding boxes (InstaOrder requirement)
        bboxes = self._find_expand_bboxes(masks)

        # Run pairwise occlusion ordering
        occ_order, depth_order = infer.infer_order_sup_occ_depth(
            self._model, image, modal, bboxes,
            pairs="all", method="InstaOrderNet_od",
            patch_or_image="resize",
            input_size=self.config.instaorder_input_size,
            disp_select_method=""
        )

        # Transpose: occ_order[i, j] = 1 means j is over i
        occ_order = occ_order.transpose()

        # Aggregate all occluders of the target
        occ_indices = np.where(occ_order[target_idx] == 1)[0]
        target_mask = masks[target_idx]
        agg_occluder = np.zeros(target_mask.shape, dtype=np.uint8)

        for idx in occ_indices:
            if idx == target_idx:
                continue
            agg_occluder = np.maximum(agg_occluder, masks[idx].astype(np.uint8))

        # Prevent occluder from containing the target mask itself
        overlap = target_mask.astype(np.uint8) + agg_occluder
        agg_occluder[overlap > 1] = 0

        # Free intermediate memory
        del modal, bboxes, occ_order
        gc.collect()

        return agg_occluder

    def handle_boundary_occlusion(self, visible_mask: np.ndarray,
                                  occ_mask: np.ndarray) -> tuple[np.ndarray, set]:
        """
        Paper Eq. 3: Expand occluder mask along image edges where target touches boundary.

        If the target object contacts image boundaries, the image frame itself
        acts as an occluder. We expand M_occ along those edges.

        Args:
            visible_mask: H×W uint8 — target's visible mask
            occ_mask: H×W uint8 — current occluder mask

        Returns:
            (updated_occ_mask, sides_touched)
        """
        gap = self.config.boundary_gap_pixels
        sides_touched = self._check_touch_boundary(visible_mask, gap)
        return occ_mask, sides_touched

    def update_canvas_boundaries(self, occ_mask_canvas: np.ndarray,
                                 amodal_mask: np.ndarray,
                                 crop_x_min: int, crop_x_max: int,
                                 crop_y_min: int, crop_y_max: int) -> np.ndarray:
        """
        Paper §3.2: After re-segmentation, expand canvas occluder mask
        along edges where amodal segmentation touches crop boundaries.
        """
        sides = self._check_touch_boundary(amodal_mask, gap_pixels=10)

        if "top" in sides:
            occ_mask_canvas[:crop_x_min + 5, :] = 1
        if "bottom" in sides:
            occ_mask_canvas[crop_x_max - 5:, :] = 1
        if "left" in sides:
            occ_mask_canvas[:, :crop_y_min + 5] = 1
        if "right" in sides:
            occ_mask_canvas[:, crop_y_max - 5:] = 1

        return occ_mask_canvas

    # ── Helper Methods ───────────────────────────────────────────────────

    def _find_expand_bboxes(self, masks: np.ndarray) -> np.ndarray:
        """
        Compute expanded bounding boxes for InstaOrder.
        InstaOrder requires expanded square boxes centered on each mask.
        """
        bboxes = np.zeros((len(masks), 4))
        for i, mask in enumerate(masks):
            m = mask.copy()
            m[m > 0] = 1
            x_arr, y_arr = np.where(m == 1)
            if len(x_arr) == 0:
                continue
            x_min, x_max = x_arr.min(), x_arr.max()
            y_min, y_max = y_arr.min(), y_arr.max()
            w = y_max - y_min
            h = x_max - x_min
            bboxes[i] = [y_min, x_min, w, h]

        # Expand to squares
        new_bboxes = []
        for bbox in bboxes:
            cx = bbox[0] + bbox[2] / 2.0
            cy = bbox[1] + bbox[3] / 2.0
            size = max(np.sqrt(bbox[2] * bbox[3] * 3.0), bbox[2] * 1.1, bbox[3] * 1.1)
            new_bboxes.append([int(cx - size / 2), int(cy - size / 2), int(size), int(size)])
        return np.array(new_bboxes)

    @staticmethod
    def _check_touch_boundary(mask: np.ndarray, gap_pixels: int = 10) -> set:
        """Check which image edges the mask touches."""
        H, W = mask.shape[:2]
        m = mask.copy()
        m[m > 0] = 1
        x_arr, y_arr = np.where(m == 1)
        if len(x_arr) == 0:
            return set()

        x_min, x_max = x_arr.min(), x_arr.max()
        y_min, y_max = y_arr.min(), y_arr.max()

        sides = set()
        if x_max >= H - gap_pixels:
            sides.add("bottom")
        if x_min <= gap_pixels:
            sides.add("top")
        if y_max >= W - gap_pixels:
            sides.add("right")
        if y_min <= gap_pixels:
            sides.add("left")
        return sides
