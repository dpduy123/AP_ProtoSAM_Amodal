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
        run_command("cd Grounded-Segment-Anything && pip install -e ./segment_anything")
        run_command("cd Grounded-Segment-Anything && export AM_I_DOCKER=False && export BUILD_WITH_CUDA=0 && pip install --no-build-isolation -e ./GroundingDINO")
        
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
    # If LISA exists but was patched with the old destructive attention_mask fix,
    # re-clone it to get a clean copy.
    if os.path.exists("LISA"):
        # Check if the destructive patch was applied
        llava_arch = "LISA/model/llava/model/llava_arch.py"
        if os.path.exists(llava_arch):
            with open(llava_arch, 'r') as f:
                content = f.read()
            if 'attention_mask' not in content or 'torch.ones' not in content:
                print("[Setup] Detected old destructive LISA patch. Re-cloning fresh LISA...")
                run_command("rm -rf LISA")

    if not os.path.exists("LISA"):
        run_command("git clone https://github.com/dvlab-research/LISA.git")
        # Replace LISA app.py with the paper's modified version
        # (adds pred_mask return at line 310 and gr.Numpy output at line 322)
        run_command("wget -O LISA/app.py https://raw.githubusercontent.com/saraao/amodal/main/LISA/app.py")
    
    # Apply safe patches (AutoConfig, MPT import, model path — but NOT attention_mask)
    run_command("python scripts/patch_lisa.py")

    # 6. Patch InstaOrder for numpy 1.24 (np.int deprecated but not yet removed)
    instaorder_inference = "InstaOrder/inference.py"
    if os.path.exists(instaorder_inference):
        run_command(f"sed -i 's/np\\.int\\b/int/g' {instaorder_inference}")
        print("[Setup] Patched InstaOrder np.int -> int")

    print("\n=== Setup Complete ===")
    print("To run LISA server:")
    print("  cd LISA && python app.py --version xinlai/LISA-13B-llama2-v1 --precision fp16 --load_in_4bit &")
    print("Then in a separate notebook cell, run the pipeline.")

if __name__ == "__main__":
    setup()
