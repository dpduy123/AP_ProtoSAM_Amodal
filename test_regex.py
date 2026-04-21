import re
content = """        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels"""

new_content = re.sub(
    r"if\s*\(\s*past_key_values is not None\s*and vision_tower is not None",
    "if (attention_mask is not None and past_key_values is not None and vision_tower is not None",
    content
)
print("--- NEW CONTENT ---")
print(new_content)
