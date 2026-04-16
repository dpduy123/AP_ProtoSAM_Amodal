import sys
import os
import torch
import numpy as np
from PIL import Image
import cv2

# Add BOTH Pix2Gestalt root and inner repo to python path
sys.path.insert(0, os.path.join(os.getcwd(), 'pix2gestalt'))
sys.path.insert(0, os.path.join(os.getcwd(), 'pix2gestalt', 'pix2gestalt'))

try:
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"[AmodalShapePredictor] Core libraries missing: {e}")
    IMPORT_SUCCESS = False

class Pix2GestaltPredictor:
    def __init__(self, ckpt_path: str = "ckpt/epoch=000005.ckpt", device: str = "cuda"):
        """
        Loads the Pix2Gestalt amodal shape predictor from a raw PyTorch Lightning checkpoint.
        Requires ~24GB VRAM.
        """
        self.device = device
        self.model = None
        print(f"[AmodalShapePredictor] Loading Pix2Gestalt Checkpoint from {ckpt_path}...")
        
        if not os.path.exists(ckpt_path):
            print(f"[AmodalShapePredictor] Warning: {ckpt_path} not found. Did you run the wget command?")
            print("[AmodalShapePredictor] Falling back to Heuristic Predictor.")
            return
            
        if not IMPORT_SUCCESS:
            print("[AmodalShapePredictor] Warning: LDM or OmegaConf not installed.")
            print("[AmodalShapePredictor] Falling back to Heuristic Predictor.")
            return

        try:
            config = OmegaConf.load("pix2gestalt/pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml")
            model = instantiate_from_config(config.model)
            
            # Load raw 15.5GB weights
            # Disable weights_only because Lightning 1.x checkpoints contain pickled ModelCheckpoint objects
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict, strict=False)
            self.model = model.to(self.device).eval()
            print("[AmodalShapePredictor] Pix2Gestalt loaded successfully.")
        except Exception as e:
            print(f"[AmodalShapePredictor] Warning: Failed to initialize network: {e}")
            print("[AmodalShapePredictor] Falling back to Heuristic Predictor.")
            self.model = None

    def predict_full_shape(self, image: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HxWx3 uint8 RGB image
            visible_mask: HxW bool array of the visible part
        Returns:
            amodal_mask: HxW bool array of the predicted whole object shape
        """
        if self.model is None:
            return self._heuristic_fallback(visible_mask)

        # 1. Run full Pix2Gestalt Inference
        print("[AmodalShapePredictor] Synthesizing amodal object with Pix2Gestalt LDM...")
        
        try:
            sys.path.insert(0, os.path.join(os.getcwd(), 'pix2gestalt', 'pix2gestalt'))
            from inference import run_pix2gestalt
            
            pil_image = Image.fromarray(image).convert("RGB")
            pil_mask = Image.fromarray((visible_mask * 255).astype(np.uint8)).convert("L")
            
            # The library typically returns a list of result PIL images
            with torch.inference_mode():
                result_pil = run_pix2gestalt(
                    self.model, 
                    pil_image, 
                    [pil_mask], 
                    device=self.device
                )[0]
                
            # 2. Extract Shape from Synthesized Output
            result_np = np.array(result_pil)
            bg_color = np.median([result_np[0,0], result_np[0,-1], result_np[-1,0], result_np[-1,-1]], axis=0)
            diff = np.linalg.norm(result_np - bg_color, axis=2)
            
            amodal_mask = (diff > 15).astype(np.uint8)
            
            kernel = np.ones((5, 5), np.uint8)
            amodal_mask = cv2.morphologyEx(amodal_mask, cv2.MORPH_CLOSE, kernel)
            amodal_mask = cv2.morphologyEx(amodal_mask, cv2.MORPH_OPEN, kernel)
            
            amodal_mask = amodal_mask.astype(bool) | visible_mask.astype(bool)
            H, W = image.shape[:2]
            amodal_mask = cv2.resize((amodal_mask*255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            return amodal_mask > 127
            
        except Exception as e:
            print(f"[AmodalShapePredictor] Error during inference: {e}. Falling back to Heuristic.")
            return self._heuristic_fallback(visible_mask)

    def _heuristic_fallback(self, visible_mask: np.ndarray) -> np.ndarray:
        print("[AmodalShapePredictor] Expanding visible mask using heuristics...")
        amodal_mask = (visible_mask.copy() * 255).astype(np.uint8)
        
        kernel = np.ones((40, 40), np.uint8)
        amodal_mask = cv2.dilate(amodal_mask, kernel, iterations=1)
        
        ys, xs = np.where(visible_mask)
        if len(ys) > 0:
            height = ys.max() - ys.min()
            bottom_y = ys.max()
            extend_to = min(amodal_mask.shape[0], int(bottom_y + height * 0.4))
            x_min, x_max = xs.min(), xs.max()
            amodal_mask[bottom_y:extend_to, x_min:x_max] = 255
            
        return amodal_mask > 127
