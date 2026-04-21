import re
from pathlib import Path

path = Path("LISA/model/LISA.py")
if not path.exists():
    print("No file")
else:
    content = path.read_text()
    if "use_cache=False" not in content:
        content = re.sub(
            r"(return_dict_in_generate=True,)",
            r"\1\n                use_cache=False,",
            content
        )
        print("Updated")
        path.write_text(content)
