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
        run_command("cd Grounded-Segment-Anything && pip install -q -e .")
        run_command("cd Grounded-Segment-Anything && pip install -q -e segment_anything")
        run_command("cd Grounded-Segment-Anything && pip install -q -e GroundingDINO")
        
        # Download checkpoints
        run_command("wget -P Grounded-Segment-Anything https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        run_command("wget -P Grounded-Segment-Anything https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")

    # 3. RAM++ (recognize-anything)
    if not os.path.exists("recognize-anything"):
        run_command("git clone https://github.com/xinyu1205/recognize-anything.git")
        run_command("cd recognize-anything && pip install -q -e .")
        
        # Download checkpoint
        run_command("wget -P recognize-anything https://github.com/xinyu1205/recognize-anything/releases/download/v1.0/ram_plus_swin_large_14m.pth")

    # 4. InstaOrder
    if not os.path.exists("InstaOrder"):
        run_command("git clone https://github.com/POSTECH-CVLab/InstaOrder")
        # Ensure checkpoint dir exists
        os.makedirs("InstaOrder/InstaOrder_ckpt", exist_ok=True)
        # Placeholder for downloading InstaOrder weights
        # Note: Users might need to download manually via Google Drive link usually provided by POSTECH-CVLab.
        print("\n[NOTE] Please ensure InstaOrder_InstaOrderNet_od.pth.tar is placed in InstaOrder/InstaOrder_ckpt/")

    # 5. LISA (VLM Segmenter)
    if not os.path.exists("LISA"):
        run_command("git clone https://github.com/dvlab-research/LISA.git")
        run_command("cd LISA && pip install -r requirements.txt")
        # Replace LISA app.py with the paper's modified version
        run_command("wget -O LISA/app.py https://raw.githubusercontent.com/saraao/amodal/main/LISA/app.py")

    print("\n=== Setup Complete ===")
    print("To run LISA server in background (requires ~26GB VRAM alone):")
    print("  cd LISA && python app.py &")
    print("Then in a separate notebook cell, load your masks or run pipeline.")

if __name__ == "__main__":
    setup()
