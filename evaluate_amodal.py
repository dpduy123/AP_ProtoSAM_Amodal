import os
import json
import torch
import numpy as np
import cv2
import gzip
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from amodal_completer import AmodalCompleter
from metrics_utils import get_amodal_metrics, calculate_iou, calculate_lpips, calculate_clip_score, calculate_ssim

class AmodalEvaluator:
    def __init__(self, device="cuda", completer=None):
        """
        Args:
            device:    "cuda" or "cpu"
            completer: optional pre-loaded AmodalCompleter — pass this to avoid
                       reloading ~25GB of model weights between notebook cells.
                       If None, a new AmodalCompleter is constructed (which will
                       still reuse class-level singletons if already loaded).
        """
        self.completer = completer if completer is not None else AmodalCompleter(device=device)
        self._owns_completer = completer is None
        self.results = []

    def cleanup(self):
        """
        Free VRAM. Only tears down the underlying models if this evaluator
        constructed its own completer; otherwise leaves it alone so the caller
        can keep reusing it.
        """
        if hasattr(self, 'completer'):
            if self._owns_completer:
                self.completer.cleanup()
            del self.completer
        
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Evaluator] GPU Memory cleared.")

    def evaluate_cocoa(self, ann_file, img_dir, limit=100):
        """
        Evaluate on the custom COCOA subset JSON.
        """
        from pycocotools import mask as mask_utils
        
        print(f"[Evaluator] Loading subset annotations from {ann_file}...")
        
        # Support for .gz files to prevent sync issues
        if ann_file.endswith(".gz"):
            with gzip.open(ann_file, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(ann_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        anns = data.get('annotations', [])
        if limit:
            anns = anns[:limit]

        print(f"[Evaluator] Evaluating COCOA on {len(anns)} objects...")

        for ann in tqdm(anns):
            img_path = os.path.join(img_dir, ann['filename'])
            
            if not os.path.exists(img_path):
                # Try cleaning filename if it has ./
                img_path = os.path.join(img_dir, ann['filename'].replace("./", ""))
                if not os.path.exists(img_path):
                    continue
                
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # 1. Decode Modal Mask (Visible)
            # COCOA usually provides visible_mask in 'visible_mask' field as RLE
            if 'visible_mask' in ann:
                visible_mask = mask_utils.decode(ann['visible_mask']).astype(bool)
            else:
                # Fallback to standard COCO 'segmentation' if it's the visible one
                visible_mask = self._decode_any_mask(ann.get('segmentation'), h, w)
            
            # 2. Decode Amodal Mask (Full Shape)
            # In full COCOA, the 'segmentation' field is the AMODAL shape
            gt_amodal_mask = self._decode_any_mask(ann.get('segmentation'), h, w)
            
            # Run Pipeline
            try:
                output = self.completer.complete(image, visible_mask, all_masks=[])
                pred_amodal_mask = output['amodal_mask']
                pred_image = output['inpainted_rgba'][:,:,:3]
                
                # Calculate Metrics
                metrics = get_amodal_metrics(pred_amodal_mask, gt_amodal_mask, visible_mask)
                
                # Appearance Metrics
                metrics['LPIPS'] = calculate_lpips(pred_image, image)
                metrics['SSIM'] = calculate_ssim(pred_image, image)
                metrics['filename'] = ann['filename']
                
                self.results.append(metrics)
            except Exception as e:
                print(f"Error processing {ann['filename']}: {e}")

        self.save_results("cocoa_subset_results.csv")

    def _decode_any_mask(self, seg, h, w):
        from pycocotools import mask as mask_utils
        if not seg:
            return np.zeros((h, w), dtype=bool)
        
        if isinstance(seg, list):
            # Polygon format
            rles = mask_utils.frPyObjects(seg, h, w)
            return mask_utils.decode(rles).any(axis=2).astype(bool)
        elif isinstance(seg, dict) and 'counts' in seg:
            # RLE format
            return mask_utils.decode(seg).astype(bool)
        return np.zeros((h, w), dtype=bool)

    def evaluate_huggingface(self, dataset_name="shunk031/COCOA", split="validation", limit=100):
        """
        Evaluate directly from a Hugging Face dataset.
        Solves the storage issue by streaming/caching efficiently.
        """
        print(f"[Evaluator] Loading HF Dataset: {dataset_name} ({split})...")
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        if limit:
            # Take a subset for evaluation
            indices = range(min(limit, len(ds)))
            ds = ds.select(indices)

        print(f"[Evaluator] Starting evaluation on {len(ds)} samples...")

        for i, sample in enumerate(tqdm(ds)):
            image_pil = sample['image'].convert("RGB")
            image_np = np.array(image_pil)
            
            # Extract masks from HF dataset structure (Aligning with shunk031/COCOA)
            # Both modal_seg and amodal_seg are stored as PIL images in this dataset
            visible_mask = np.array(sample['modal_seg']).astype(bool)
            gt_amodal_mask = np.array(sample['amodal_seg']).astype(bool)
            
            # If mask is RGB, convert to single channel
            if len(visible_mask.shape) == 3:
                visible_mask = visible_mask[:, :, 0]
            if len(gt_amodal_mask.shape) == 3:
                gt_amodal_mask = gt_amodal_mask[:, :, 0]
            
            try:
                # Run Pipeline
                output = self.completer.complete(image_np, visible_mask, all_masks=[])
                pred_amodal_mask = output['amodal_mask']
                pred_image = output['inpainted_rgba'][:,:,:3] # Get RGB from RGBA
                
                # 1. Shape Metrics
                shape_metrics = get_amodal_metrics(pred_amodal_mask, gt_amodal_mask, visible_mask)
                
                # 2. Appearance Metrics (The ones from your CVPR table)
                lpips_val = calculate_lpips(pred_image, image_np)
                ssim_val = calculate_ssim(pred_image, image_np)
                
                # Full result row
                result = {
                    "id": i,
                    "mIoU": shape_metrics['mIoU'],
                    "oIoU": shape_metrics['oIoU'],
                    "LPIPS": lpips_val,
                    "SSIM": ssim_val
                }
                self.results.append(result)
                
            except Exception as e:
                print(f"Error on sample {i}: {e}")

        self.save_results("hf_eval_results.csv")

    def save_results(self, filename):
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\n[Evaluator] Results saved to {filename}")
        print(f"Mean mIoU: {df['mIoU'].mean():.4f}")
        print(f"Mean oIoU: {df['oIoU'].mean():.4f}")

if __name__ == "__main__":
    # Example usage (adjust paths for your Colab environment)
    # evaluator = AmodalEvaluator()
    # evaluator.evaluate_cocoa("path/to/cocoa.json", "path/to/images", limit=50)
    pass
