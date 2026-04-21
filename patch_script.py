import re
from pathlib import Path

def patch_file():
    p = Path("LISA/model/llava/model/llava_arch.py")
    if not p.exists():
        return
    content = p.read_text()
    
    # Let's bypass the attention_mask reassignment entirely!
    new_content = re.sub(
        r"attention_mask = torch\.ones\([^)]+\)",
        "pass  # attention_mask = torch.ones removed to avoid NoneType errors",
        content
    )
    # Also catch any multi-line ones
    new_content = re.sub(
        r"attention_mask = torch\.ones\([\s\S]*?device=attention_mask\.device,\s*\)",
        "pass  # attention_mask = torch.ones removed",
        new_content
    )
    p.write_text(new_content)
