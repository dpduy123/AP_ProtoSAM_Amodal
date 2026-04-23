"""
pipeline/pipeline.py — End-to-End Orchestrator

Links all components of the CVPR 2025 Open-World Amodal Appearance Completion pipeline.
"""

import os
import numpy as np
from PIL import Image
import torch
import cv2

from pipeline.config import PipelineConfig
from pipeline.vlm_segmenter import VLMSegmenter
from pipeline.scene_analyzer import SceneAnalyzer
from pipeline.occlusion_analyzer import OcclusionAnalyzer
from pipeline.prompt_selector import PromptSelector
from pipeline.iterative_inpainter import IterativeInpainter
from pipeline.blender import Blender


class AmodalPipeline:
    """
    Main orchestrator.
    
    Assumes:
    1. LISA server is running or masks are already cached.
    2. RAM, SAM, Grounding DINO, InstaOrder, SD2 are prepared.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        self.vlm = VLMSegmenter(self.config.lisa_server_url, self.config.lisa_output_path)
        self.scene = SceneAnalyzer(config)
        self.occlusion = OcclusionAnalyzer(config)
        self.prompt = PromptSelector(config.clip_model_name, config.device)
        self.inpainter = IterativeInpainter(config)
        self.blender = Blender()

    def run(self, image_path: str, text_query: str) -> dict:
        """
        Run the full paper pipeline on a single image.
        """
        img_pil = Image.open(image_path).convert("RGB")
        img_array = np.array(img_pil)
        
        print(f"\n[Pipeline] Starting: {image_path} | Query: '{text_query}'")
        
        # Stage 1: Get visible mask (cached or from LISA)
        # Note: If LISA server isn't running and it isn't cached, this fails.
        visible_mask = self.vlm.segment(image_path, text_query, use_cache=True)
        
        if visible_mask is None or visible_mask.sum() == 0:
            raise ValueError("Target object not found by VLM segmenter.")
           
        # Stage 2: Scene Analyzer (RAM++ tags, G-DINO boxes, SAM masks, BG segments)
        scene = self.scene.analyze(img_array, img_pil, target_class=text_query)
        if len(scene.masks) == 0:
             raise ValueError("Scene analyzer found no objects.")
             
        # Stage 3: Prompt Selector
        prompt_text = self.prompt.select(img_pil, visible_mask, scene.tags, text_query)
        
        # Build all masks pool for occlusion
        # Pool = object masks + visible target mask + background segments
        pool = list(scene.masks)
        target_idx = len(pool)
        pool.append(visible_mask)
        pool.extend(scene.background_segments)
        pool_array = np.array(pool)
        
        # Occlusion Analysis
        occ_mask = self.occlusion.analyze(img_array, pool_array, target_idx)
        occ_mask, _ = self.occlusion.handle_boundary_occlusion(visible_mask, occ_mask)
        
        # Memory cleanup before SD2
        self.scene.unload_models()
        self.prompt._model = None
        torch.cuda.empty_cache()
        
        # Stage 5: Iterative Inpainting (BYPASSED AS REQUESTED)
        print("\n[Pipeline] Bypassing SD2 Inpainting, outputting raw Amodal Mask (Visible + Occluded)...")
        amodal_mask = visible_mask | occ_mask
        
        # Create a visual representation (white mask on black background)
        amodal_vis = np.zeros_like(img_array)
        amodal_vis[amodal_mask] = [255, 255, 255] # White for amodal shape
        
        # Overlay original visible textures for better visualization
        amodal_vis[visible_mask] = img_array[visible_mask]
        
        blended_crop = Image.fromarray(amodal_vis).convert("RGBA")
        
        inpaint_result = {
            "blended_crop": blended_crop,
            "iter_count": 0
        }
            
        # Clean up heavy models
        self.inpainter.unload_model()
        self.occlusion.unload_model()
        
        # Stage 6: Alpha Blending
        amodal_pil = inpaint_result["amodal_completion"]
        amodal_mask = inpaint_result["amodal_mask"]
        cx_min, cx_max, cy_min, cy_max = inpaint_result["crop_region"][:4]
        
        # Reconstruct full-size orig RGBA
        crop_orig = img_array[cx_min:cx_max, cy_min:cy_max]
        crop_vis_mask = visible_mask[cx_min:cx_max, cy_min:cy_max]
        orig_rgba = np.zeros((*crop_orig.shape[:2], 4), dtype=np.uint8)
        orig_rgba[:,:,:3] = crop_orig
        orig_rgba[:,:,3] = crop_vis_mask.astype(np.uint8) * 255
        
        # Reconstruct amodal RGBA
        amodal_array = np.array(amodal_pil)
        amodal_rgba = np.zeros((amodal_array.shape[0], amodal_array.shape[1], 4), dtype=np.uint8)
        amodal_rgba[:,:,:3] = amodal_array
        amodal_rgba[:,:,3] = amodal_mask.astype(np.uint8) * 255
        
        blended_crop = self.blender.blend(orig_rgba, amodal_rgba)
        blended_pil = Image.fromarray(blended_crop)
        
        # Save or return
        return {
            "blended_crop": blended_pil,
            "amodal_mask_crop": amodal_mask, 
            "prompt_used": prompt_text,
            "iter_count": inpaint_result["iter_count"]
        }
