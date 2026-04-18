"""
pipeline/prompt_selector.py — Stage 4: CLIP-based Prompt Selection

Paper §3.3 (Eq. 4): Select optimal inpainting prompt by matching the visible
target object against candidate descriptors using CLIP similarity.

  P = argmax_{ti ∈ T ∪ {Q}} CLIP(I_target, ti)

Reference:
  CLIP: Learning Transferable Visual Models from Natural Language Supervision (PMLR 2021)
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional


class PromptSelector:
    """
    Selects the best text descriptor for the target object by comparing
    CLIP image features of the visible region against candidate labels.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self._model = None
        self._preprocess = None
        self._model_name = model_name

    def load_model(self):
        """Load CLIP model."""
        if self._model is not None:
            return
        import clip
        print(f"[PromptSelector] Loading CLIP {self._model_name}...")
        self._model, self._preprocess = clip.load(self._model_name, self.device)
        print("[PromptSelector] CLIP loaded.")

    def select(self, image_pil: Image.Image, visible_mask: np.ndarray,
               tags: list[str], text_query: str) -> str:
        """
        Paper Eq. 4: Select best inpainting prompt from T ∪ {Q}.

        Args:
            image_pil: Original image
            visible_mask: H×W bool — visible region of target
            tags: list of auto-detected class tags from RAM++ (T)
            text_query: User's original text query (Q)

        Returns:
            Best matching descriptor string
        """
        import clip

        if self._model is None:
            self.load_model()

        # Create masked image: only show target object, rest black
        image_np = np.array(image_pil)
        masked = np.zeros_like(image_np)
        mask_bool = visible_mask.astype(bool)
        for c in range(3):
            masked[:, :, c] = image_np[:, :, c] * mask_bool
        masked_pil = Image.fromarray(masked.astype(np.uint8))

        # Candidates: T ∪ {Q}
        candidates = tags + [text_query]
        # Remove empty strings and duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            c_clean = c.strip()
            if c_clean and c_clean.lower() not in seen:
                seen.add(c_clean.lower())
                unique_candidates.append(c_clean)
        candidates = unique_candidates

        if not candidates:
            return text_query or "object"

        # CLIP encode
        image_input = self._preprocess(masked_pil).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(
            [f"a photo of a {c}" for c in candidates]
        ).to(self.device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            text_features = self._model.encode_text(text_inputs)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        best_idx = similarity[0].argmax().item()
        best_prompt = candidates[best_idx]

        print(f"[PromptSelector] Selected prompt: '{best_prompt}' "
              f"(score={similarity[0][best_idx]:.4f})")
        return best_prompt
