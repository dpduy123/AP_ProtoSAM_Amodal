"""
pipeline/amodal_completer.py

Open-World Amodal Appearance Completion pipeline.
Given:
  - RGB image
  - visible mask of the target object (from SAM)
  - all other masks in the scene
  - optional text query

Returns: RGBA image (H×W×4) where the 4th channel is the amodal mask.

Pipeline (follows CVPR 2025 paper):
  1. Occlusion analysis  → identify which masks occlude the target
  2. Prompt selection    → CLIP-based best descriptor
  3. Image preparation   → isolate target on clean background
  4. Iterative inpainting → Stable Diffusion v2 inpainting
  5. Alpha blending      → seamless merge + RGBA output
"""

import numpy as np
import torch
import cv2
from PIL import Image
from typing import Optional


class AmodalCompleter:

    def __init__(
        self,
        inpainting_model_id: str = "sd2-community/stable-diffusion-2-inpainting",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None
        self._clip_model = None
        self._clip_processor = None
        self._instaorder = None
        self._load_models(inpainting_model_id, clip_model_id)

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_models(self, inpainting_model_id: str, clip_model_id: str):
        print("[AmodalCompleter] Loading Stable Diffusion inpainting model...")
        from diffusers import StableDiffusionInpaintPipeline

        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            inpainting_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self._pipe.to(self.device)
        self._pipe.set_progress_bar_config(disable=True)

        # ── VRAM optimizations (Colab T4/A100) ──
        if self.device == "cuda":
            self._pipe.enable_attention_slicing("max")   # ~30% less VRAM
            self._pipe.enable_vae_slicing()               # VAE decode in slices
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
                print("[AmodalCompleter] xformers enabled (faster + less VRAM)")
            except Exception:
                print("[AmodalCompleter] xformers not available, using attention slicing")

        print("[AmodalCompleter] Loading CLIP model...")
        from transformers import CLIPModel, CLIPProcessor

        clip_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._clip_model = CLIPModel.from_pretrained(
            clip_model_id, torch_dtype=clip_dtype
        ).to(self.device)
        self._clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

        print("[AmodalCompleter] Models loaded.")

    # ── Main entry point ───────────────────────────────────────────────────

    def complete(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        all_masks: list[dict],
        text_query: str = "",
        max_iter: int = 3,
        epsilon: float = 0.01,
    ) -> np.ndarray:
        """
        Args:
            image:        H×W×3 uint8 RGB
            visible_mask: H×W bool — visible region of target object
            all_masks:    list of SAM mask dicts (other objects in scene)
            text_query:   optional text description of the target
            max_iter:     max inpainting iterations
            epsilon:      convergence threshold for occluder mask delta

        Returns:
            H×W×4 uint8 RGBA — completed object with transparent background
        """
        H, W = image.shape[:2]

        # Step 1: Build occluder mask
        occluder_mask = self._build_occluder_mask(
            image, visible_mask, all_masks, H, W
        )

        # Step 2: Select best inpainting prompt
        prompt = self._select_prompt(image, visible_mask, text_query)

        # Step 3: Prepare inpainting image (target on clean bg)
        target_img, background = self._prepare_target_image(image, visible_mask)

        # Step 4: Iterative inpainting
        amodal_mask = visible_mask.copy().astype(bool)
        inpaint_img = target_img.copy()
        prev_occ_mask = occluder_mask.copy()

        for iteration in range(max_iter):
            inpaint_img, newly_reconstructed = self._inpaint_step(
                inpaint_img, occluder_mask, prompt, H, W
            )

            # Update amodal mask
            amodal_mask = amodal_mask | newly_reconstructed

            # Update occluder mask — remove regions now reconstructed
            occluder_mask = occluder_mask & (~newly_reconstructed)

            # Check convergence
            delta = np.sum(np.abs(occluder_mask.astype(float) - prev_occ_mask.astype(float)))
            normalized_delta = delta / (H * W)
            if normalized_delta < epsilon or not occluder_mask.any():
                break
            prev_occ_mask = occluder_mask.copy()

        # Step 5: Alpha blending
        blended_rgb = self._alpha_blend(image, inpaint_img, visible_mask, amodal_mask)

        # Step 6: Build RGBA output
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3] = blended_rgb
        rgba[:, :, 3] = (amodal_mask * 255).astype(np.uint8)

        return rgba

    # ── Step 1: Occlusion analysis ─────────────────────────────────────────

    def _build_occluder_mask(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        all_masks: list[dict],
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Identify which segments occlude the target object.

        Key insight: SAM produces non-overlapping masks, so adjacent masks
        (touching the target boundary) are likely occluders. We dilate the
        target mask to detect these adjacencies.
        """
        occluder = np.zeros((H, W), dtype=bool)

        # Dilate the visible mask to find nearby/adjacent masks
        # This is critical because SAM masks typically DON'T overlap
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated_target = cv2.dilate(
            visible_mask.astype(np.uint8), dilate_kernel
        ).astype(bool)
        target_border_zone = dilated_target & (~visible_mask)

        # Also get the convex hull of the visible mask to estimate full extent
        target_hull = self._convex_hull_mask(visible_mask, H, W)
        # The "missing" region: inside hull but not in visible mask
        estimated_hidden = target_hull & (~visible_mask)

        for i, m in enumerate(all_masks):
            seg = m["segmentation"].astype(bool)

            # Skip if this is the target mask itself
            if np.array_equal(seg, visible_mask):
                continue

            # Check 1: Direct overlap (rare with SAM, but possible)
            direct_overlap = seg & visible_mask
            # Check 2: Adjacent — mask touches the dilated target boundary
            adjacent_overlap = seg & target_border_zone
            # Check 3: Mask covers the estimated hidden region
            hidden_overlap = seg & estimated_hidden

            if not direct_overlap.any() and not adjacent_overlap.any() and not hidden_overlap.any():
                continue

            # Score how likely this mask is an occluder
            adjacency_score = adjacent_overlap.sum() / max(target_border_zone.sum(), 1)
            hidden_score = hidden_overlap.sum() / max(estimated_hidden.sum(), 1)

            # If the mask significantly overlaps with where we expect hidden parts
            # OR it's strongly adjacent to the target boundary → it's an occluder
            if hidden_score > 0.05 or adjacency_score > 0.03 or direct_overlap.any():
                try:
                    is_occluder = self._check_occlusion_order(image, visible_mask, seg)
                except Exception:
                    # Default: assume adjacent masks in the hidden region are occluders
                    is_occluder = hidden_score > 0.02 or adjacency_score > 0.02
                if is_occluder:
                    occluder |= seg

        # Handle boundary cases: expand along image edges
        occluder = self._handle_boundary_occlusion(visible_mask, occluder, H, W)

        # The actual region to inpaint: estimated hidden area covered by occluders
        # (not the full occluder — we only want to reveal what's behind it)
        inpaint_region = estimated_hidden | (occluder & target_hull)

        # Also add a border expansion to ensure smooth completion
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        inpaint_region = cv2.dilate(
            inpaint_region.astype(np.uint8), expand_kernel
        ).astype(bool)

        # Don't inpaint inside the already-visible mask
        inpaint_region = inpaint_region & (~visible_mask)

        print(f"[AmodalCompleter] Occluder masks found: {occluder.any()}")
        print(f"[AmodalCompleter] Estimated hidden region: {estimated_hidden.sum()} px")
        print(f"[AmodalCompleter] Inpaint region: {inpaint_region.sum()} px")

        return inpaint_region

    def _convex_hull_mask(
        self, mask: np.ndarray, H: int, W: int
    ) -> np.ndarray:
        """
        Compute the convex hull of a binary mask.
        The convex hull estimates the full amodal extent of the object.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return mask.copy()

        # Merge all contours and find convex hull
        all_points = np.concatenate(contours)
        hull = cv2.convexHull(all_points)

        hull_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        return hull_mask.astype(bool)

    def _check_occlusion_order(
        self,
        image: np.ndarray,
        target_mask: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> bool:
        """
        Check if candidate_mask occludes target_mask.
        Uses InstaOrderNet if available, otherwise falls back to heuristic.
        """
        try:
            return self._instaorder_check(image, target_mask, candidate_mask)
        except Exception:
            return self._heuristic_occlusion_check(target_mask, candidate_mask)

    def _instaorder_check(self, image, target_mask, candidate_mask) -> bool:
        """
        Use InstaOrderNet for pairwise occlusion order prediction.
        InstaOrderNet: https://github.com/HnKnA/InstaOrderNet
        """
        # InstaOrderNet takes (image, mask1, mask2) and returns occlusion probability
        # Install: pip install instaordernet
        import instaordernet
        model = instaordernet.load_model()  # cached after first call
        prob = model.predict_occlusion(image, target_mask, candidate_mask)
        return prob > 0.5  # candidate occludes target if prob > 0.5

    def _heuristic_occlusion_check(
        self, target_mask: np.ndarray, candidate_mask: np.ndarray
    ) -> bool:
        """
        Fallback heuristic: a segment likely occludes the target if:
        - It is adjacent to the target's boundary
        - It covers area where the target's convex hull extends
        - Smaller objects in front of larger ones are likely occluders
        """
        target_area = target_mask.sum()
        candidate_area = candidate_mask.sum()

        # Dilate target to check adjacency
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_target = cv2.dilate(target_mask.astype(np.uint8), kernel).astype(bool)
        adjacency = (candidate_mask & dilated_target & ~target_mask).sum()

        # If there's significant adjacency, likely an occluder
        if adjacency > 50:
            return True

        # Erode target mask to get interior
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        target_interior = cv2.erode(target_mask.astype(np.uint8), erode_kernel).astype(bool)
        target_boundary = target_mask & (~target_interior)

        # Overlap with boundary (not interior) → likely occlusion
        boundary_overlap = candidate_mask & target_boundary
        interior_overlap = candidate_mask & target_interior
        return boundary_overlap.sum() > interior_overlap.sum() * 0.3

    def _handle_boundary_occlusion(
        self,
        visible_mask: np.ndarray,
        occluder_mask: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        If the target touches image boundaries, dilate the occluder mask
        along those edges (the image frame itself is an occluder).
        """
        edges_touched = {
            "top": visible_mask[0, :].any(),
            "bottom": visible_mask[-1, :].any(),
            "left": visible_mask[:, 0].any(),
            "right": visible_mask[:, -1].any(),
        }

        if not any(edges_touched.values()):
            return occluder_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_occ = cv2.dilate(occluder_mask.astype(np.uint8), kernel).astype(bool)

        # Only add dilation near the touched edges
        edge_region = np.zeros((H, W), dtype=bool)
        margin = 30
        if edges_touched["top"]:    edge_region[:margin, :] = True
        if edges_touched["bottom"]: edge_region[-margin:, :] = True
        if edges_touched["left"]:   edge_region[:, :margin] = True
        if edges_touched["right"]:  edge_region[:, -margin:] = True

        occluder_mask = occluder_mask | (dilated_occ & edge_region)
        return occluder_mask

    # ── Step 2: Prompt selection ───────────────────────────────────────────

    def _select_prompt(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
        text_query: str,
    ) -> str:
        """
        Use CLIP to select the best text descriptor for the visible object region.
        Combines auto-generated tags with the user's text query.
        """
        # Crop visible region
        ys, xs = np.where(visible_mask)
        if len(xs) == 0:
            return text_query or "object"

        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = image[y1:y2+1, x1:x2+1].copy()

        # Mask out non-target pixels
        crop_mask = visible_mask[y1:y2+1, x1:x2+1]
        crop[~crop_mask] = 128  # neutral gray background

        crop_pil = Image.fromarray(crop)

        # Candidate descriptions: user query + generic tags
        candidates = [
            text_query,
            "object", "item", "thing",
            "person", "animal", "vehicle", "furniture",
            "foreground object", "occluded object",
        ]
        candidates = [c for c in candidates if c.strip()]

        # CLIP similarity
        inputs = self._clip_processor(
            text=candidates,
            images=[crop_pil] * len(candidates),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._clip_model(**inputs)
            # Image-text similarity for the crop vs each candidate
            image_feats = outputs.image_embeds[0:1]  # (1, D)
            text_feats = outputs.text_embeds          # (N, D)
            sims = (image_feats @ text_feats.T).squeeze(0)
            best_idx = sims.argmax().item()

        return candidates[best_idx]

    # ── Step 3: Target image preparation ──────────────────────────────────

    def _prepare_target_image(
        self,
        image: np.ndarray,
        visible_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Isolate the target on a neutral background.
        Background pixels are replaced with the image's median color
        to provide a natural context-free canvas for inpainting.
        """
        background = np.full_like(image, image.reshape(-1, 3).mean(axis=0).astype(np.uint8))

        target_img = image.copy().astype(float)
        mask_3ch = visible_mask[:, :, np.newaxis]
        target_img = (target_img * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

        return target_img, background

    # ── Step 4: Inpainting step ────────────────────────────────────────────

    def _inpaint_step(
        self,
        image: np.ndarray,
        occluder_mask: np.ndarray,
        prompt: str,
        H: int,
        W: int,
        inpaint_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one iteration of Stable Diffusion inpainting.

        Args:
            image:         current inpainting canvas (H×W×3)
            occluder_mask: H×W bool — regions to inpaint
            prompt:        text conditioning for the inpainting
            H, W:          original image dimensions
            inpaint_size:  SD inference resolution

        Returns:
            (inpainted_image H×W×3, newly_reconstructed_mask H×W bool)
        """
        if not occluder_mask.any():
            return image, np.zeros((H, W), dtype=bool)

        # Resize to SD input resolution
        pil_image = Image.fromarray(image).resize((inpaint_size, inpaint_size), Image.LANCZOS)
        pil_mask = Image.fromarray((occluder_mask * 255).astype(np.uint8)).resize(
            (inpaint_size, inpaint_size), Image.NEAREST
        )

        with torch.inference_mode():
            result = self._pipe(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=inpaint_size,
                width=inpaint_size,
            ).images[0]

        # Resize back to original resolution
        result_np = np.array(result.resize((W, H), Image.LANCZOS))

        # Newly reconstructed = pixels that were in occluder_mask and now have content
        newly_reconstructed = occluder_mask.copy()

        # Blend: only replace occluder pixels
        blended = image.copy()
        blended[occluder_mask] = result_np[occluder_mask]

        return blended, newly_reconstructed

    # ── Step 5: Alpha blending ─────────────────────────────────────────────

    def _alpha_blend(
        self,
        original_image: np.ndarray,
        inpainted_image: np.ndarray,
        visible_mask: np.ndarray,
        amodal_mask: np.ndarray,
        transition_px: int = 12,
    ) -> np.ndarray:
        """
        Blend original visible pixels with inpainted reconstructed pixels.
        Creates a smooth transition at the boundary of the visible region.

        Alpha = 1.0  inside visible_mask (use original pixels)
        Alpha = 0→1  in transition zone (blend)
        Alpha = 1.0  in reconstructed region (use inpainted pixels)
        """
        # Distance transform from visible mask boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (transition_px * 2 + 1,) * 2)
        eroded = cv2.erode(visible_mask.astype(np.uint8), kernel)
        transition_zone = visible_mask.astype(np.uint8) - eroded

        # Build alpha map
        dist = cv2.distanceTransform(visible_mask.astype(np.uint8), cv2.DIST_L2, 5)
        alpha_map = np.clip(dist / transition_px, 0, 1)
        alpha_map[~visible_mask] = 0
        alpha_3ch = alpha_map[:, :, np.newaxis]

        # Blend: original inside visible, inpainted outside
        blended = (
            original_image.astype(float) * alpha_3ch +
            inpainted_image.astype(float) * (1 - alpha_3ch)
        ).astype(np.uint8)

        # Only keep amodal region
        result = np.zeros_like(original_image)
        result[amodal_mask] = blended[amodal_mask]

        return result
