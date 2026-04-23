import re

content = """
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            new_attn_mask_pad_right = torch.full(
"""

pattern = r"([ \t]*)attention_mask\s*=\s*torch\.ones\([\s\S]*?device\s*=\s*attention_mask\.device,?\s*\)"
new_content = re.sub(pattern, r"\1pass", content)
print(new_content)
