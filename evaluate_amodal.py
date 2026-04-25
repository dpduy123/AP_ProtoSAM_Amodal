import os
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from amodal_completer import AmodalCompleter
from metrics_utils import get_amodal_metrics, calculate_iou, calculate_lpips, calculate_clip_score, calculate_ssim

class AmodalEvaluator:
    def __init__(self, device="cuda"):
        self.completer = AmodalCompleter(device=device)
        self.results = []

    def cleanup(self):
        """
        Force cleanup of the underlying models to free VRAM.
        """
        if hasattr(self, 'completer'):
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
        Evaluate on COCO Amodal dataset.
        Requires pycocotools: pip install pycocotools
        """
        try:
            from pycocotools.coco import COCO
        except ImportError:
            print("Error: pycocotools not installed. Run 'pip install pycocotools'")
            return

        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        if limit:
            img_ids = img_ids[:limit]

        print(f"[Evaluator] Evaluating COCOA on {len(img_ids)} images...")

        for img_id in tqdm(img_ids):
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            for ann in anns:
                if 'amodal_mask' not in ann or 'mask' not in ann:
                    continue
                
                # Decode masks
                visible_mask = coco.annToMask(ann).astype(bool)
                # COCOA usually stores amodal mask in 'amodal_seg' or similar, 
                # depends on the specific version of the dataset.
                # Here we assume a standard binary format.
                # (You might need to adjust this depending on your JSON structure)
                gt_amodal_mask = self._decode_amodal_mask(ann, img_info['height'], img_info['width'], coco)
                
                # Run Pipeline
                try:
                    output = self.completer.complete(image, visible_mask, all_masks=[])
                    pred_amodal_mask = output['amodal_mask']
                    
                    # Calculate Metrics
                    metrics = get_amodal_metrics(pred_amodal_mask, gt_amodal_mask, visible_mask)
                    metrics['img_id'] = img_id
                    metrics['ann_id'] = ann['id']
                    
                    self.results.append(metrics)
                except Exception as e:
                    print(f"Error processing ann {ann['id']}: {e}")

        self.save_results("cocoa_results.csv")

    def _decode_amodal_mask(self, ann, h, w, coco):
        # Simplified: in some versions cocoa has 'amodal_mask' as RLE
        # This part depends on the specific COCOA variant you are using
        if 'amodal_seg' in ann:
            # Handle RLE or Polygon
             return coco.annToMask(ann) # Placeholder: you should use the amodal field
        return coco.annToMask(ann) # Fallback to modal

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
