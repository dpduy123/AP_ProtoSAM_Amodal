"""
pipeline/scene_analyzer.py — Stage 2: Scene Understanding

Paper §3.1: Auto-detect all objects + background segments in the scene.

Pipeline:
  1. RAM++ (open-set tagging) → extract class tags T = {t1, t2, ...}
  2. Grounding DINO (open-set detection) → bounding boxes per tag
  3. SAM (Segment Anything) → pixel-precise masks from boxes
  4. Background Segments (Eq. 1) → morphology on unsegmented regions

Reference:
  - RAM++: Open-Set Image Tagging (arXiv 2023)
  - Grounding DINO: Open-Set Object Detection (ECCV 2024)
  - SAM: Segment Anything (CVPR 2023)
"""

import sys
import os
import gc
import math
import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional
from dataclasses import dataclass

from pipeline.config import PipelineConfig


@dataclass
class SceneResult:
    """Container for scene analysis output."""
    image: np.ndarray           # H×W×3 uint8 RGB
    masks: np.ndarray           # (N, H, W) bool — all object masks
    class_names: list[str]      # class label per mask
    pred_scores: list[float]    # confidence per mask
    tags: list[str]             # RAM++ auto-detected tags
    background_segments: list[np.ndarray]  # list of H×W bool — background segment masks


class SceneAnalyzer:
    """
    Comprehensive scene understanding using RAM++ + Grounding DINO + SAM.
    Produces all object masks AND background segments for occlusion analysis.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self._gdino_model = None
        self._sam_predictor = None
        self._ram_model = None
        self._ram_transform = None
        self._loaded = False

    def load_models(self):
        """Load all models. Call explicitly to control VRAM timing."""
        if self._loaded:
            return

        print("[SceneAnalyzer] Loading models...")
        self._load_grounding_dino()
        self._load_ram()
        # SAM is loaded per-call to avoid keeping the predictor in memory
        print("[SceneAnalyzer] All models loaded.")
        self._loaded = True

    def unload_models(self):
        """Free GPU memory."""
        del self._gdino_model, self._ram_model, self._ram_transform
        self._gdino_model = None
        self._ram_model = None
        self._ram_transform = None
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
        print("[SceneAnalyzer] Models unloaded.")

    # ── Main entry point ─────────────────────────────────────────────────

    def analyze(self, image: np.ndarray, image_pil: Image.Image,
                target_class: Optional[str] = None) -> SceneResult:
        """
        Run full scene analysis.

        Args:
            image: H×W×3 uint8 RGB numpy array
            image_pil: PIL Image (same content)
            target_class: Optional class name to ensure it's detected

        Returns:
            SceneResult with masks, class_names, tags, background_segments
        """
        if not self._loaded:
            self.load_models()

        # 1. RAM++ → auto-detect tags
        tags = self._run_ram(image_pil)
        tags.append("background")
        print(f"[SceneAnalyzer] RAM++ tags: {tags[:10]}{'...' if len(tags) > 10 else ''}")

        # 2. Grounding DINO → bounding boxes
        img_tensor = self._transform_image(image_pil)
        classes_str = ". ".join(tags)
        boxes_filt, pred_phrases = self._run_gdino(img_tensor, classes_str)

        # Ensure target class is detected (paper: run separate query if not found)
        if target_class:
            target_found = any(target_class.lower() in p.lower() for p in pred_phrases)
            if not target_found:
                target_boxes, target_phrases = self._run_gdino(img_tensor, target_class)
                if len(target_boxes) > 0:
                    boxes_filt = torch.cat((boxes_filt, target_boxes), dim=0)
                    pred_phrases = list(pred_phrases) + list(target_phrases)

        # 3. SAM → masks from boxes
        if len(boxes_filt) == 0:
            print("[SceneAnalyzer] Warning: No objects detected!")
            return SceneResult(
                image=image, masks=np.array([]),
                class_names=[], pred_scores=[],
                tags=tags, background_segments=[]
            )

        img_result, masks = self._run_sam(image_pil, boxes_filt, pred_phrases)
        if masks is None:
            print("[SceneAnalyzer] Warning: SAM failed to produce masks!")
            return SceneResult(
                image=image, masks=np.array([]),
                class_names=[], pred_scores=[],
                tags=tags, background_segments=[]
            )

        # Parse class names and scores from pred_phrases
        class_names, pred_scores = self._parse_phrases(pred_phrases)

        # 4. Background segments (Paper Eq. 1)
        bg_segments = self._extract_background_segments(masks, image.shape[:2])
        print(f"[SceneAnalyzer] Detected {len(masks)} objects, {len(bg_segments)} background segments")

        return SceneResult(
            image=img_result if img_result is not None else image,
            masks=masks,
            class_names=class_names,
            pred_scores=pred_scores,
            tags=tags,
            background_segments=bg_segments,
        )

    def re_segment(self, image_pil: Image.Image, classes: list[str],
                   target_class: str) -> tuple[np.ndarray, list[str], list[float]]:
        """
        Re-segment an inpainted image during iterative inpainting.
        Used inside the iteration loop to detect amodal mask expansion.
        """
        img_tensor = self._transform_image(image_pil)
        img_array = np.array(image_pil)

        boxes_filt, pred_phrases = self._run_gdino(img_tensor, ". ".join(classes), target_class=target_class)

        # Ensure target class query
        target_found = any(target_class.lower() in p.lower() for p in pred_phrases)
        if not target_found:
            target_boxes, target_phrases = self._run_gdino(img_tensor, target_class)
            if len(target_boxes) > 0:
                boxes_filt = torch.cat((boxes_filt, target_boxes), dim=0)
                pred_phrases = list(pred_phrases) + list(target_phrases)

        if len(boxes_filt) == 0:
            return None, [], []

        _, masks = self._run_sam(image_pil, boxes_filt, pred_phrases)
        if masks is None:
            return None, [], []

        class_names, pred_scores = self._parse_phrases(pred_phrases)
        return masks, class_names, pred_scores

    # ── Background Segments (Paper Eq. 1) ────────────────────────────────

    def _extract_background_segments(self, masks: np.ndarray, shape: tuple) -> list[np.ndarray]:
        """
        Paper Eq. 1: B_j = Morph(I - ∪ S_i)

        Identifies unsegmented regions, applies erosion to sharpen boundaries,
        then finds connected components as separate background segments.
        """
        from skimage.morphology import erosion, disk

        H, W = shape
        combined = np.zeros((H, W), dtype=np.uint8)
        for mask in masks:
            combined = np.maximum(combined, mask.astype(np.uint8))

        # Unsegmented = everything not covered by any object mask
        unsegmented = (1 - combined).astype(np.uint8)

        # Erosion: sharpen boundaries, separate loosely connected areas
        selem = disk(2)
        eroded = erosion(unsegmented, selem)

        # Find connected components
        contours, _ = cv2.findContours(
            eroded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        segments = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Skip tiny noise regions
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, -1)
                segments.append(mask.astype(bool))

        return segments

    # ── Model Loading ────────────────────────────────────────────────────

    def _load_grounding_dino(self):
        """Load Grounding DINO model."""
        sys.path.append("Grounded-Segment-Anything")
        sys.path.append("Grounded-Segment-Anything/GroundingDINO")

        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict

        args = SLConfig.fromfile(self.config.gdino_config)
        args.device = self.device
        model = build_model(args)
        ckpt = torch.load(self.config.gdino_ckpt, map_location="cpu")
        model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        model.eval()
        self._gdino_model = model
        print("[SceneAnalyzer] Grounding DINO loaded.")

    def _load_ram(self):
        """Load RAM++ open-set tagging model."""
        from ram.models import ram_plus
        from ram import get_transform

        model = ram_plus(
            pretrained=self.config.ram_ckpt,
            image_size=self.config.ram_image_size,
            vit="swin_l"
        )
        model.class_threshold = torch.ones(model.num_class) * 0.6
        model.eval()
        model = model.to(self.device)
        self._ram_model = model
        self._ram_transform = get_transform(image_size=self.config.ram_image_size)
        print("[SceneAnalyzer] RAM++ loaded.")

    # ── Inference Methods ────────────────────────────────────────────────

    def _run_ram(self, image_pil: Image.Image) -> list[str]:
        """Run RAM++ to get auto-detected tags."""
        from ram import inference_ram as inference

        image = self._ram_transform(
            image_pil.resize((self.config.ram_image_size, self.config.ram_image_size))
        ).unsqueeze(0).to(self.device)

        res = inference(image, self._ram_model)
        tags = [t.strip() for t in res[0].split("|")]
        return tags

    def _run_gdino(self, img_tensor: torch.Tensor, caption: str,
                   target_class: Optional[str] = None) -> tuple:
        """Run Grounding DINO for open-set object detection."""
        from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap
        import re

        model = self._gdino_model.to(self.device)
        img = img_tensor.to(self.device)

        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."

        with torch.no_grad():
            outputs = model(img[None], captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

        filt_mask = logits.max(dim=1)[0] > self.config.gdino_box_thresh
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        pred_phrases = []
        tokenizer = model.tokenizer
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > self.config.gdino_text_thresh, tokenizer(caption), tokenizer
            )
            pred_phrases.append(f"{pred_phrase}({logit.max().item():.4f})")

        return boxes_filt, pred_phrases

    def _run_sam(self, image_pil: Image.Image, boxes_filt: torch.Tensor,
                 pred_phrases: list = None) -> tuple:
        """Run SAM to produce masks from bounding boxes."""
        from segment_anything import build_sam, SamPredictor

        img = np.array(image_pil)
        predictor = SamPredictor(build_sam(checkpoint=self.config.sam_ckpt).to(self.device))
        predictor.set_image(img)

        H, W = image_pil.size[1], image_pil.size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        try:
            transformed_boxes = predictor.transform.apply_boxes_torch(
                boxes_filt, img.shape[:2]
            ).to(self.device)
            masks, iou_predictions, _ = predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=transformed_boxes, multimask_output=False,
            )
        except Exception as e:
            print(f"[SceneAnalyzer] SAM prediction failed: {e}")
            return None, None

        masks = masks.cpu().numpy().squeeze(1)  # (N, H, W)

        # Free SAM predictor
        del predictor
        torch.cuda.empty_cache()

        return img, masks

    def _transform_image(self, image_pil: Image.Image) -> torch.Tensor:
        """Transform PIL image to tensor for Grounding DINO."""
        import GroundingDINO.groundingdino.datasets.transforms as T

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(image_pil, None)
        return img_tensor

    def _parse_phrases(self, pred_phrases: list[str]) -> tuple[list[str], list[float]]:
        """Parse 'classname(0.9123)' into separate lists."""
        import re
        class_names = []
        pred_scores = []
        for phrase in pred_phrases:
            parts = re.split(r"\(|\)", phrase)
            class_names.append(parts[0])
            pred_scores.append(float(parts[1]) if len(parts) > 1 else 0.0)
        return class_names, pred_scores
