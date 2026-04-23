"""
scripts/patch_lisa.py — Patch LISA repo để tương thích với transformers mới và Hugging Face Hub.

Được gọi từ colab_setup_paper.py SAU KHI git clone LISA về.
"""

import re
from pathlib import Path


def patch(filepath: str, patches: list[tuple[str, str]], label: str):
    """Apply list of (old, new) regex substitutions to a file."""
    p = Path(filepath)
    if not p.exists():
        print(f"[patch] SKIP {label}: file not found at {filepath}")
        return
    content = p.read_text()
    for old, new in patches:
        if old in content:
            content = content.replace(old, new)
            print(f"[patch] ✅ {label}: applied")
        else:
            print(f"[patch] ⚠️  {label}: pattern not found (may already be patched)")
    p.write_text(content)


def main():
    base = Path("LISA")

    # ── Patch 1: llava_llama.py — AutoConfig collision ──────────────────
    patch(
        str(base / "model/llava/model/language_model/llava_llama.py"),
        [
            (
                'AutoConfig.register("llava", LlavaConfig)',
                'AutoConfig.register("llava", LlavaConfig, exist_ok=True)'
            )
        ],
        "AutoConfig exist_ok"
    )

    # ── Patch 2: __init__.py — disable MPT import ────────────────────────
    patch(
        str(base / "model/llava/model/__init__.py"),
        [
            (
                "from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM",
                "# from .language_model.llava_mpt import LlavaMPTConfig, LlavaMPTForCausalLM  # disabled: incompatible with transformers>=4.35"
            )
        ],
        "Disable MPT import"
    )

    # ── Patch 3: hf_prefixlm_converter.py — _expand_mask removed ─────────
    patch(
        str(base / "model/llava/model/language_model/mpt/hf_prefixlm_converter.py"),
        [
            (
                "from transformers.models.bloom.modeling_bloom import (\n    _expand_mask,",
                "# _expand_mask removed in transformers>=4.35, providing stub\ndef _expand_mask(mask, dtype, tgt_len=None): return mask\n# from transformers.models.bloom.modeling_bloom import (\n#     _expand_mask,"
            ),
            # Alternative single-line form
            (
                "from transformers.models.bloom.modeling_bloom import _expand_mask",
                "# _expand_mask removed in transformers>=4.35, providing stub\ndef _expand_mask(mask, dtype, tgt_len=None): return mask"
            )
        ],
        "_expand_mask stub"
    )

    # ── Patch 4: app.py — replace local model path with HF Hub ID ────────
    app_py = base / "app.py"
    if app_py.exists():
        content = app_py.read_text()
        # Replace any variant of the local path (with single or double quotes)
        content = re.sub(
            r"""(['"])\.\/LISA-13B-llama2-v1\1""",
            '"xinlai/LISA-13B-llama2-v1"',
            content
        )
        # Also handle without leading ./
        content = re.sub(
            r"""(['"])LISA-13B-llama2-v1\1""",
            '"xinlai/LISA-13B-llama2-v1"',
            content
        )
        app_py.write_text(content)
        print("[patch] ✅ app.py: model path → xinlai/LISA-13B-llama2-v1")
    else:
        print("[patch] ⚠️  app.py: not found, skipping")

    # ── Patch 5: llava_arch.py — attention_mask fix ────────────────────────
    # NOTE: With transformers==4.31.0 (pinned in requirements_paper.txt),
    # attention_mask is NEVER None inside prepare_inputs_labels_for_multimodal.
    # The torch.ones crash only occurs with transformers>=4.36.
    # Therefore, NO patch is needed here. The previous regex that replaced
    # the attention_mask block with 'pass' DESTROYED LISA's mask quality,
    # causing blocky 32x32 patch-level outputs.
    print("[patch] ℹ️  llava_arch.py: no patch needed (transformers==4.31.0 is safe)")

    # ── Patch 6: LISA.py — fix transformers>=4.36 use_cache crash ────────
    lisa_py = base / "model/LISA.py"
    if lisa_py.exists():
        content = lisa_py.read_text()
        if "use_cache=False" not in content:
            new_content = re.sub(
                r"(return_dict_in_generate=True,?)",
                r"\1\n                use_cache=False,",
                content
            )
            if new_content != content:
                lisa_py.write_text(new_content)
                print("[patch] ✅ LISA.py: use_cache=False fix applied")
            else:
                print("[patch] ⚠️  LISA.py: return_dict_in_generate not found")


if __name__ == "__main__":
    main()
