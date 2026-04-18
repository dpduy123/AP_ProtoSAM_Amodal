"""
pipeline/blender.py — Stage 6: Alpha Blending

Paper §3.4 (Eq. 8): Final alpha blending to achieve a smooth transition between 
newly reconstructed regions and original visible regions.
"""

import cv2
import numpy as np
from PIL import Image

class Blender:
    """
    Handles alpha blending of the amodal completion into the original image.
    """
    
    def blend(self, orig_rgba: np.ndarray, amodal_rgba: np.ndarray, 
              transition_width: int = 5) -> np.ndarray:
        """
        Paper Eq 8: Alpha blend.
        
        Args:
            orig_rgba: Original image crop with visible alpha.
            amodal_rgba: SD2 inpainted crop with amodal alpha.
            transition_width: Pixels for distance transform blend.
            
        Returns:
            Blended RGBA image.
        """
        # Shrink original visible mask slightly to avoid seam artifacts
        src = self._shrink_edges(orig_rgba, shrink_amount=10)
        dst = amodal_rgba
        
        if src.shape[:2] != dst.shape[:2]:
            raise ValueError("Size mismatch in blender.")
            
        src_mask = (src[:, :, 3] > 0).astype(np.uint8)
        
        # Interior erosion
        kernel = np.ones((transition_width, transition_width), np.uint8)
        src_interior = cv2.erode(src_mask, kernel, iterations=1)
        transition = src_mask - src_interior
        
        # Distance transform for smooth weights
        dist = cv2.distanceTransform((1 - src_mask).astype(np.uint8), cv2.DIST_L2, 5)
        dist = np.clip(dist / transition_width, 0, 1)
        
        w_src = np.where(transition > 0, dist, 1.0)
        w_dst = 1.0 - w_src
        
        blended = dst.copy()
        
        # Core visible stays original
        blended[src_mask > 0] = src[src_mask > 0]
        
        # Transition area blend
        dst_alpha = dst[transition > 0, 3]
        blend_idx = dst_alpha > 0
        
        if np.any(blend_idx):
            src_trans = src[transition > 0, :3]
            dst_trans = dst[transition > 0, :3]
            w_src_t = w_src[transition > 0, np.newaxis]
            w_dst_t = w_dst[transition > 0, np.newaxis]
            
            blended[transition > 0, :3][blend_idx] = (
                w_dst_t[blend_idx] * dst_trans[blend_idx] +
                w_src_t[blend_idx] * src_trans[blend_idx]
            )
            
        no_blend = ~blend_idx
        if np.any(no_blend):
             blended[transition > 0, :3][no_blend] = src[transition > 0, :3][no_blend]
             
        # Resolve final alpha
        blended[:, :, 3] = np.maximum(src[:, :, 3], dst[:, :, 3])
        
        return blended

    def _shrink_edges(self, image: np.ndarray, shrink_amount: int) -> np.ndarray:
        """Make edges of the alpha mask transparent to reduce boundary artifacts."""
        alpha = image[:, :, 3]
        m = (alpha == 0).astype(np.uint8)
        kernel = np.ones((shrink_amount * 2 + 1,) * 2, np.uint8)
        dilated = cv2.dilate(m, kernel, iterations=1)
        
        mod = image.copy()
        mod[dilated > 0, 3] = 0
        return mod
