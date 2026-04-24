import torch
from PIL import Image
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

class VLMReasoner:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device="cuda"):
        """
        Initializes the official Qwen3-VL model using the newest architecture class.
        """
        self.device = device
        print(f"[VLMReasoner] Loading Official {model_id} (FP16)...")
        
        try:
            # Using the specific Qwen3VL class as per official documentation
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            print(f"[VLMReasoner] {model_id} loaded successfully.")
        except Exception as e:
            print(f"[VLMReasoner] Failed to load model: {e}")
            print("Tip: Ensure you ran !pip install git+https://github.com/huggingface/transformers")
            self.model = None

    @torch.no_grad()
    def reason_occlusion(self, image_np, visible_mask=None):
        if self.model is None:
            return "ERROR: Model not loaded."

        image_pil = Image.fromarray(image_np)
        
        # Standard Qwen3-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": "Describe the missing parts of the target object for inpainting prompt."}
                ],
            }
        ]

        # Official preparation method
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generating output
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        # Slicing out the prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()

    @torch.no_grad()
    def get_missing_region_boxes(self, image_np):
        """
        Uses Qwen-VL's grounding capability to find bounding boxes of occluded/missing parts.
        Returns: List of [ymin, xmin, ymax, xmax] in normalized coordinates (0-1000).
        """
        if self.model is None: return []

        image_pil = Image.fromarray(image_np)
        prompt = "Identify the locations of the hidden or missing parts of the main object in this image. Output the bounding boxes in the format [ymin, xmin, ymax, xmax]."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=200)
        output_text = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Extract coordinates using regex: e.g., [123, 456, 789, 101]
        boxes = []
        pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches = re.findall(pattern, output_text)
        
        for m in matches:
            boxes.append([int(x) for x in m])
        
        return boxes
