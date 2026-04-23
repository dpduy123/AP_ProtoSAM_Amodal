"""
pipeline/iterative_inpainter.py — Stage 5: Iterative SD2 Inpainting

Paper §3.4: Iteratively inpaint the occluded region, re-segment, and check
for remaining occlusions until stable or max iterations reached.
Includes a canvas 6x expansion to allow outpainting if the object touches borders.
"""

import math
import numpy as np
import cv2
import torch
from PIL import Image

from pipeline.config import PipelineConfig


class IterativeInpainter:
    """
    Handles the iterative masked inpainting loop (Eq. 5-7).
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = config.device
        self._model = None
        self._loaded = False

    def load_model(self):
        """Load Stable Diffusion 2 Inpainting model."""
        if self._loaded:
            return
            
        print(f"[IterativeInpainter] Loading SD2 {self.config.sd_model_id}...")
        from diffusers import StableDiffusionInpaintPipeline
        
        self._model = StableDiffusionInpaintPipeline.from_pretrained(
            self.config.sd_model_id,
            torch_dtype=torch.float16,
        )
        self._model.enable_model_cpu_offload()
        self._model.enable_attention_slicing()
        self._loaded = True
        print("[IterativeInpainter] SD2 Inpainting loaded with CPU offload.")

    def unload_model(self):
        """Free GPU memory."""
        del self._model
        self._model = None
        self._loaded = False
        torch.cuda.empty_cache()

    def inpaint(self, image: np.ndarray, visible_mask: np.ndarray, 
                occ_mask: np.ndarray, prompt: str,
                scene_analyzer, occlusion_analyzer,
                target_class: str, tags: list[str]) -> dict:
        """
        Run the iterative inpainting process.
        
        Args:
            image: Original image (H×W×3 uint8)
            visible_mask: Visible region of the target (H×W bool)
            occ_mask: Initial occluder mask (H×W bool)
            prompt: Text prompt for inpainting
            scene_analyzer: Instance of SceneAnalyzer for re-segmentation
            occlusion_analyzer: Instance of OcclusionAnalyzer for re-checking
            target_class: Original class to track
            tags: RAM++ tags for context
        """
        if not self._loaded:
            self.load_model()
            
        H, W = image.shape[:2]
        
        # Base background image for isolating the target (Eq. 5)
        # We generate a generic gray background instead of loading an external file
        bg_array = np.full((H, W, 3), 128, dtype=np.uint8)
        
        # 1. Canvas System: Create 6x expanded canvas (to allow outpainting)
        mult = self.config.canvas_multiplier
        img_canvas = self._create_canvas(image, mult, self.config.canvas_fill_value)
        valid_query_canvas = self._create_canvas(visible_mask, mult, 0)
        occ_canvas = self._create_canvas(occ_mask, mult, 0)
        outpaint_canvas = self._create_canvas(np.zeros_like(visible_mask), mult, 1)

        # Store initial states for offsets
        init_valid_query_canvas = valid_query_canvas.copy()
        init_outpaint_canvas = outpaint_canvas.copy()

        first_sd_occ_mask = None
        amodal_segmentation = None
        current_amodal_mask = visible_mask.copy()
        
        for t in range(self.config.max_iter):
            print(f"[IterativeInpainter] --- Iteration {t+1}/{self.config.max_iter} ---")
            
            # Crop to relevant region 
            crop_region = self._find_crop_region(visible_mask, init_valid_query_canvas)
            crop_x_min, crop_x_max, crop_y_min, crop_y_max, _ = crop_region
            
            crop_img = img_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max]
            crop_visible = valid_query_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max]
            crop_occ = occ_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max]
            crop_outpaint = outpaint_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max]
            
            # Form final SD occlusion mask
            sd_occ = crop_occ + crop_outpaint
            sd_occ[sd_occ > 0] = 1
            
            # Format prompt
            sd_prompt = f"a {prompt}"
            
            # Dilate mask for SD
            kernel = np.ones((3, 3), np.uint8)
            if t == 0:
                first_sd_occ_mask = sd_occ.copy()
                sd_occ = cv2.dilate(sd_occ.astype(np.uint8), np.ones((5,5), np.uint8), iterations=3)
                
                # Background swap for first iter (Eq. 5)
                # Ensure the background crop fits
                if bg_array.shape[0] < crop_img.shape[0] or bg_array.shape[1] < crop_img.shape[1]:
                    print("[IterativeInpainter] Warning: Clean bg image is smaller than crop, padding bg.")
                    pad_h = max(0, crop_img.shape[0] - bg_array.shape[0])
                    pad_w = max(0, crop_img.shape[1] - bg_array.shape[1])
                    bg_array = np.pad(bg_array, ((0, pad_h), (0, pad_w), (0,0)), mode='reflect')
                
                bg_crop = bg_array[:crop_img.shape[0], :crop_img.shape[1]]
                bg_mask = (1 - crop_visible).astype(bool)
                crop_img[bg_mask] = bg_crop[bg_mask]
            else:
                # Lock occluder mask to intersection of current and first iteration's occluder mask
                if first_sd_occ_mask is not None:
                     sd_occ = np.logical_and(sd_occ, first_sd_occ_mask).astype(np.uint8)
                sd_occ = cv2.dilate(sd_occ.astype(np.uint8), kernel, iterations=1)
                
            if not np.any(sd_occ):
                print("[IterativeInpainter] Occluder mask is empty, stopping.")
                break
                
            # Run SD2 Inpainting
            sd_img_pil = Image.fromarray(crop_img).resize((self.config.sd_target_size,) * 2, Image.LANCZOS)
            sd_occ_pil = Image.fromarray(sd_occ * 255).convert("L").resize((self.config.sd_target_size,) * 2, Image.NEAREST)
            
            with torch.no_grad():
                result_pil = self._model(
                    image=sd_img_pil,
                    mask_image=sd_occ_pil,
                    prompt=sd_prompt
                ).images[0]
                
            # Resize back to crop size
            result_pil = result_pil.resize((crop_img.shape[1], crop_img.shape[0]))
            result_array = np.array(result_pil)
            
            # --- Re-segmentation & Re-analysis ---
            new_masks, new_class_names, _ = scene_analyzer.re_segment(result_pil, tags, target_class)
            
            if not new_masks or len(new_masks) == 0:
                 print("[IterativeInpainter] No object found in re-segmentation. Stopping.")
                 break
                 
            # Find the amodal mask matching the original crop_visible
            amodal_idx, amodal_mask = self._filter_amodal_segmentation(crop_visible, new_masks)
            amodal_segmentation = amodal_mask
            
            # Recheck occlusions on the INPAINTED crop
            new_occ_masks = occlusion_analyzer.analyze(result_array, new_masks, amodal_idx)
            # Prevent new occluders from overlapping with the amodal mask
            new_occ_masks[np.logical_and(amodal_mask > 0, new_occ_masks > 0)] = 0
            
            # Update the global canvases
            img_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = result_array
            valid_query_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = amodal_mask
            occ_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = new_occ_masks
            outpaint_canvas[crop_x_min:crop_x_max, crop_y_min:crop_y_max] = 0
            
            # Boundary expansion
            occ_canvas = occlusion_analyzer.update_canvas_boundaries(
                occ_canvas, amodal_mask, crop_x_min, crop_x_max, crop_y_min, crop_y_max
            )
            
            # Termination Check (Eq. 7)
            # In paper, check L1 norm of occluder mask change
            occ_sum = new_occ_masks.sum()
            sides = occlusion_analyzer._check_touch_boundary(amodal_mask)
            if occ_sum == 0 and len(sides) == 0:
                print("[IterativeInpainter] Stable! No more occluders.")
                break
                
        # After loop
        if amodal_segmentation is None:
             print("[IterativeInpainter] Failed to get amodal mask.")
             return None
             
        # Extract the final bounding piece matching the original image dimensions
        # BUT shifted by the outpainting amodal mask offsets.
        x_off, y_off = self._compute_offset(valid_query_canvas, init_outpaint_canvas, amodal_segmentation)
        
        # We need the final result_pil, but padded/cropped properly to the original image dimensions
        # The paper code essentially saves the raw crop and processes it later in blender.
        
        return {
            "amodal_completion": result_pil,  # The final crop PIL
            "amodal_mask": amodal_segmentation, # Mask specific to the crop
            "x_offset": x_off,
            "y_offset": y_off,
            "crop_region": crop_region, # (crop_x_min, crop_x_max, crop_y_min, crop_y_max)
            "iter_count": t + 1
        }

    # --- Helper Methods ---

    def _create_canvas(self, input_arr: np.ndarray, mult: int, val: int) -> np.ndarray:
        """Place input in center of a mult x mult larger canvas."""
        shape = list(input_arr.shape)
        orig_h, orig_w = shape[0], shape[1]
        shape[0] = int(orig_h * mult)
        shape[1] = int(orig_w * mult)
        
        if val > 0:
            canvas = np.ones(shape, dtype=input_arr.dtype) * val
        else:
            canvas = np.zeros(shape, dtype=input_arr.dtype)
            
        start_h = (shape[0] // 2) - (orig_h // 2)
        start_w = (shape[1] // 2) - (orig_w // 2)
        
        canvas[start_h:start_h+orig_h, start_w:start_w+orig_w] = input_arr
        return canvas

    def _find_crop_region(self, orig_mask: np.ndarray, canvas_mask: np.ndarray):
        """Find the tightest crop around the query mask on the canvas, plus padding."""
        h, w = canvas_mask.shape[:2]
        pad = self.config.pad_pixels
        buf = self.config.crop_buffer
        
        m = canvas_mask.copy()
        m[m > 0] = 1
        x_arr, y_arr = np.where(m == 1)
        if len(x_arr) == 0:
            return 0, h, 0, w, max(h, w)
            
        x_min, x_max = x_arr.min(), x_arr.max()
        y_min, y_max = y_arr.min(), y_arr.max()
        
        cx_min = max(0, x_min - buf)
        cx_max = min(h, x_max + buf)
        cy_min = max(0, y_min - buf)
        cy_max = min(w, y_max + buf)
        
        # Check original mask boundaries to see if outpainting padding is needed
        orig_x, orig_y = np.where((orig_mask > 0).astype(int) == 1)
        if len(orig_x) > 0:
            o_xmin, o_xmax = orig_x.min(), orig_x.max()
            o_ymin, o_ymax = orig_y.min(), orig_y.max()
            orig_h, orig_w = orig_mask.shape[:2]
            
            gap = self.config.boundary_gap_pixels
            if o_xmin <= gap: cx_min = max(0, cx_min - pad)
            if o_xmax >= orig_h - gap: cx_max = min(h, cx_max + pad)
            if o_ymin <= gap: cy_min = max(0, cy_min - pad)
            if o_ymax >= orig_w - gap: cy_max = min(w, cy_max + pad)
            
        # Target square size
        crop_h, crop_w = cx_max - cx_min, cy_max - cy_min
        tar = max(crop_h, crop_w)
        
        # Make square (simplified logic compared to full paper, expands equally)
        if crop_w < tar:
            diff = tar - crop_w
            cy_min = max(0, cy_min - diff // 2)
            cy_max = min(w, cy_max + (diff - diff // 2))
        if crop_h < tar:
            diff = tar - crop_h
            cx_min = max(0, cx_min - diff // 2)
            cx_max = min(h, cx_max + (diff - diff // 2))
            
        return cx_min, cx_max, cy_min, cy_max, tar

    def _filter_amodal_segmentation(self, crop_visible: np.ndarray, masks: list[np.ndarray]):
        """Match the new segmentation mask that best overlaps with the original visible mask."""
        best_iou = 0
        best_i = 0
        best_mask = crop_visible.copy()
        
        for i, m in enumerate(masks):
            overlap = np.logical_and(crop_visible > 0, m > 0).sum()
            union = np.logical_or(crop_visible > 0, m > 0).sum()
            iou = overlap / (union + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_i = i
                best_mask = m
                
        return best_i, best_mask

    def _compute_offset(self, valid_canvas: np.ndarray, outpaint_canvas: np.ndarray, amodal_seg: np.ndarray):
        """Compute the offset of the amodal mask relative to original image center."""
        q_x, q_y = np.where(valid_canvas == 1)
        o_x, o_y = np.where(outpaint_canvas == 0)
        a_x, a_y = np.where(amodal_seg == 1)
        
        if len(q_x) == 0 or len(o_x) == 0 or len(a_x) == 0:
            return 0, 0
            
        return int(q_x.min() - a_x.min() - o_x.min()), int(q_y.min() - a_y.min() - o_y.min())
