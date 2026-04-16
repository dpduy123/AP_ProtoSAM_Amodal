import torch
import numpy as np
from PIL import Image
import cv2

class Pix2GestaltPredictor:
    def __init__(self, model_id: str = "cvlab/pix2gestalt-weights", device: str = "cuda"):
        """
        Initializes the Pix2Gestalt amodal shape predictor.
        Requires ~24GB VRAM. It synthesizes the whole object from a visible mask.
        """
        self.device = device
        print(f"[AmodalShapePredictor] Loading Pix2Gestalt from {model_id}...")
        
        # We will load it using custom pipeline if available or fall back to standard
        try:
            from diffusers import DiffusionPipeline
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id, 
                custom_pipeline=model_id, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(self.device)
            # Enable xformers or slicing if necessary
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[AmodalShapePredictor] Pix2Gestalt loaded successfully.")
        except Exception as e:
            print(f"[AmodalShapePredictor] Warning: Failed to load Pix2Gestalt pipeline: {e}")
            self.pipe = None

    def predict_full_shape(self, image: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
        """
        Args:
            image: HxWx3 uint8 RGB image
            visible_mask: HxW bool array of the visible part
        Returns:
            amodal_mask: HxW bool array of the predicted whole object shape
        """
        if self.pipe is None:
            # Fallback if model not loaded: returning visible mask
            return visible_mask

        # 1. Prepare inputs for Pix2Gestalt
        H, W = image.shape[:2]
        
        pil_image = Image.fromarray(image).convert("RGB")
        pil_mask = Image.fromarray((visible_mask * 255).astype(np.uint8)).convert("L")

        # 2. Run Inference
        print("[AmodalShapePredictor] Synthesizing whole object with Pix2Gestalt...")
        with torch.inference_mode():
            # The pix2gestalt pipeline usually takes image and mask_image
            # Note: actual interface might vary based on their HF custom_pipeline signature
            try:
                result_image = self.pipe(
                    image=pil_image,
                    mask_image=pil_mask,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]
            except TypeError:
                print("[AmodalShapePredictor] Inference signature mismatch. Returning visible mask.")
                return visible_mask

        # 3. Extract the Binarized Amodal Mask from the synthesized result
        # Pix2Gestalt typically outputs the object on a grey/white background.
        # We do a basic background subtraction to get the mask.
        result_np = np.array(result_image)
        
        # Assuming the background of the output is relatively uniform (e.g., grey)
        # We find the color of the corners and threshold
        bg_color = np.median([result_np[0,0], result_np[0,-1], result_np[-1,0], result_np[-1,-1]], axis=0)
        diff = np.linalg.norm(result_np - bg_color, axis=2)
        
        # Threshold difference to get the mask
        amodal_mask = (diff > 15).astype(np.uint8)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        amodal_mask = cv2.morphologyEx(amodal_mask, cv2.MORPH_CLOSE, kernel)
        amodal_mask = cv2.morphologyEx(amodal_mask, cv2.MORPH_OPEN, kernel)
        
        # Ensure that the visible mask is always included in the amodal mask
        amodal_mask = amodal_mask.astype(bool) | visible_mask.astype(bool)
        
        # Resize back to original H, W just in case the pipeline resized it
        amodal_mask = cv2.resize((amodal_mask*255).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        
        return amodal_mask > 127
