"""
pipeline — Open-World Amodal Appearance Completion (CVPR 2025)

Faithful reproduction of:
  Ao, Jiang, Ke, Ehinger. "Open-World Amodal Appearance Completion." CVPR 2025.
  https://github.com/saraao/amodal

Stages:
  1. VLM Segmenter (LISA)    → visible mask from text query
  2. Scene Analyzer           → all objects + background segments
  3. Occlusion Analyzer       → InstaOrderNet occlusion ordering
  4. Prompt Selector (CLIP)   → best inpainting prompt
  5. Iterative Inpainter      → SD2 inpainting loop
  6. Blender                  → alpha blend → RGBA output
"""

from pipeline.config import PipelineConfig
