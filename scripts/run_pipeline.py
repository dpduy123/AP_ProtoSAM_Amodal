"""
scripts/run_pipeline.py — CLI to run the CVPR 2025 pipeline.

Usage:
  python scripts/run_pipeline.py --image test.jpg --query "polar bear" --mode vlm
  python scripts/run_pipeline.py --image test.jpg --query "polar bear" --mode full
"""

import argparse
import sys
import os

# Add parent dir to path so we can import pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="CVPR 2025 Open-World Amodal Appearance Completion")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--query", type=str, required=True, help="Text query identifying the target object")
    parser.add_argument("--mode", type=str, choices=["vlm", "full"], default="full", 
                        help="'vlm' runs only LISA (stage 1) to cache masks. 'full' runs the entire pipeline.")
    
    args = parser.parse_args()
    
    config = PipelineConfig()
    
    if args.mode == "vlm":
        print("=== Running VLM Segmenter (LISA) standalone ===")
        from pipeline.vlm_segmenter import VLMSegmenter
        vlm = VLMSegmenter(config.lisa_server_url, config.lisa_output_path)
        # Attempt to cache the mask
        vlm.segment_batch([args.image], [args.query])
        
    elif args.mode == "full":
        print("=== Running Full Pipeline ===")
        from pipeline.pipeline import AmodalPipeline
        pipeline = AmodalPipeline(config)
        
        result = pipeline.run(args.image, args.query)
        
        # Save results
        basename = os.path.splitext(os.path.basename(args.image))[0]
        out_dir = os.path.join(config.output_dir, f"{basename}_pipeline_output")
        os.makedirs(out_dir, exist_ok=True)
        
        result["blended_crop"].save(os.path.join(out_dir, "amodal_rgba.png"))
        
        print(f"Pipeline finished in {result['iter_count']} iterations.")
        print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    main()
