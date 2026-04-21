"""
scripts/colab_setup_paper.py — Setup script for Google Colab.

Sets up all dependencies and downloads checkpoints (~30GB) for the
CVPR 2025 Open-World Amodal Appearance Completion pipeline.
"""

import os
import subprocess
import sys


def run_command(command: str):
    """Run a shell command and stream output."""
    print(f"=== Running: {command} ===")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.decode().strip())
    rc = process.poll()
    if rc != 0:
        print(f"Command failed with return code {rc}")
        sys.exit(rc)


def setup():
    # 1. Base requirements
    run_command("pip install -r requirements_paper.txt")
    
    # 2. Grounding DINO + SAM
    if not os.path.exists("Grounded-Segment-Anything"):
        run_command("git clone https://github.com/IDEA-Research/Grounded-Segment-Anything")
        run_command("cd Grounded-Segment-Anything && pip install ./segment_anything")
        run_command("cd Grounded-Segment-Anything && BUILD_WITH_CUDA=0 pip install ./GroundingDINO")
        
        # Download checkpoints
        run_command("wget -P Grounded-Segment-Anything https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        run_command("wget -P Grounded-Segment-Anything https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")

    # 3. RAM++ (recognize-anything)
    if not os.path.exists("recognize-anything"):
        run_command("git clone https://github.com/xinyu1205/recognize-anything.git")
        run_command("cd recognize-anything && pip install .")
        
        # Download checkpoint
        run_command("wget -P recognize-anything https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth")

    # 4. InstaOrder
    if not os.path.exists("InstaOrder"):
        run_command("git clone https://github.com/POSTECH-CVLab/InstaOrder")
        # Ensure checkpoint dir exists
        os.makedirs("InstaOrder/InstaOrder_ckpt", exist_ok=True)
        # Download InstaOrder weights from their official Google Drive link
        print("[Setup] Downloading InstaOrder weights (this might take a while, ~3.5GB)...")
        run_command("pip install -q gdown")
        run_command("gdown --id 1_GEmCmofLSkJZnidfp4vsQb2Nqq5aqBU -O InstaOrder_ckpt.zip")
        print("[Setup] Unzipping InstaOrder weights...")
        run_command("unzip -q InstaOrder_ckpt.zip")
        run_command("mv InstaOrder_ckpt/InstaOrder_InstaOrderNet_od.pth.tar InstaOrder/InstaOrder_ckpt/")
        # Clean up
        run_command("rm -rf InstaOrder_ckpt.zip InstaOrder_ckpt")

    # 5. LISA (VLM Segmenter)
    if not os.path.exists("LISA"):
        run_command("git clone https://github.com/dvlab-research/LISA.git")
        # Fix collision with newer transformers which already have 'llava'
        run_command("sed -i 's/AutoConfig.register(\"llava\", LlavaConfig)/AutoConfig.register(\"llava\", LlavaConfig, exist_ok=True)/' LISA/model/llava/model/language_model/llava_llama.py")
        # Fix ImportError: cannot import name '_expand_mask' from 'transformers.models.bloom.modeling_bloom'
        # We comment out the MPT import in __init__.py as we focus on Llama models
        run_command("sed -i 's/from .language_model.llava_mpt/# from .language_model.llava_mpt/' LISA/model/llava/model/__init__.py")
        # AND we patch the problematic file directly to be safe
        run_command("sed -i 's/from transformers.models.bloom.modeling_bloom import _expand_mask/def _expand_mask(*args, **kwargs): pass\\n# from transformers.models.bloom.modeling_bloom import _expand_mask/' LISA/model/llava/model/language_model/mpt/hf_prefixlm_converter.py")
        # Replace LISA app.py with the paper's modified version
        run_command("wget -O LISA/app.py https://raw.githubusercontent.com/saraao/amodal/main/LISA/app.py")
        # Fix: Redirect local model path to Hugging Face ID so it downloads automatically
        run_command("sed -i 's|\"./LISA-13B-llama2-v1\"|\"xinlai/LISA-13B-llama2-v1\"|g' LISA/app.py")

    print("\n=== Setup Complete ===")
    print("To run LISA server in background (requires ~26GB VRAM alone):")
    print("  cd LISA && python app.py &")
    print("Then in a separate notebook cell, load your masks or run pipeline.")

if __name__ == "__main__":
    setup()
