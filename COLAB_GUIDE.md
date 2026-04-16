# 🚀 Hướng dẫn chạy AmodalSeg trên Google Colab Pro

## Yêu cầu

- Tài khoản **Google Colab Pro** (hoặc Pro+)
- GPU: **T4 (16GB)** hoặc **A100 (40GB)**

---

## Bước 1: Tạo notebook & chọn GPU

1. Vào [Google Colab](https://colab.research.google.com/) → **New Notebook**
2. Chọn **Runtime → Change runtime type**
3. Chọn:
   - **Hardware accelerator**: `GPU`
   - **GPU type**: `T4` hoặc `A100` (nếu có)
4. Nhấn **Save**

---

## Bước 2: Clone repo & cài đặt

### Cell 1 — Clone project

```python
!git clone https://github.com/dpduy123/AP_ProtoSAM_Amodal.git
%cd AP_ProtoSAM_Amodal
```

### Cell 2 — Kiểm tra GPU

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 3 — Cài dependencies

> ⚠️ **KHÔNG** cài lại torch/torchvision — Colab đã có sẵn đúng version.

```python
# CHỈ cài thêm packages thiếu (KHÔNG pip install torch)
!pip install -q torch==2.10.0 torchvision
!pip install -q diffusers transformers accelerate safetensors
!pip install -q opencv-python-headless Pillow
!pip install -q fastapi uvicorn python-multipart
```

### Cell 4 — Cài SAM2

```python
# SAM2 (recommended)
!pip install -q git+https://github.com/facebookresearch/sam2.git

# Download SAM2 checkpoint (~2.4 GB)
import os
if not os.path.exists("sam2.1_hiera_large.pt"):
    !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    print("✅ SAM2 checkpoint downloaded")
else:
    print("✅ SAM2 checkpoint already exists")
```

---

## Bước 3: Chạy pipeline

### Option A — Chạy trực tiếp trong notebook (khuyến nghị)

```python
import numpy as np
from PIL import Image
from google.colab import files

# Upload ảnh
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
img = np.array(Image.open(image_path).convert("RGB"))
print(f"Image: {image_path} | Shape: {img.shape}")
```

```python
# ── Bước 1: SAM Segmentation ──
from segmenter import SAMSegmenter

segmenter = SAMSegmenter()
masks = segmenter.segment_everything(img, points_per_side=32)
print(f"✅ Found {len(masks)} masks")

# Giải phóng SAM để dành VRAM cho SD2
del segmenter
torch.cuda.empty_cache()
```

```python
# ── Xem masks ──
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, min(4, len(masks)), figsize=(16, 4))
if len(masks) == 1:
    axes = [axes]
for i, ax in enumerate(axes):
    ax.imshow(img)
    ax.imshow(masks[i]["segmentation"], alpha=0.5, cmap="jet")
    ax.set_title(f"Mask {i} | Area: {masks[i]['area']}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

```python
# ── Bước 2: Amodal Completion ──
from amodal_completer import AmodalCompleter

# ⚠️ Chọn mask_id muốn complete (xem hình ở trên)
MASK_ID = 0  # <-- thay số này

target_mask = masks[MASK_ID]["segmentation"].astype(bool)

completer = AmodalCompleter()
rgba_result = completer.complete(
    image=img,
    visible_mask=target_mask,
    all_masks=masks,
    max_iter=3,
)
print("✅ Amodal completion done!")
```

```python
# ── Xem kết quả ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img)
axes[0].set_title("Original")

axes[1].imshow(target_mask, cmap="gray")
axes[1].set_title(f"Visible Mask (#{MASK_ID})")

axes[2].imshow(rgba_result)
axes[2].set_title("Amodal Result (RGBA)")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Lưu kết quả
result_pil = Image.fromarray(rgba_result, mode="RGBA")
output_name = f"amodal_result_mask{MASK_ID}.png"
result_pil.save(output_name)
print(f"✅ Saved: {output_name}")

# Download về máy
files.download(output_name)
```

---

### Option B — Chạy FastAPI server + ngrok (truy cập UI)

```python
# Cài ngrok
!pip install -q pyngrok

from pyngrok import ngrok

# Đăng ký free token tại https://dashboard.ngrok.com/signup
ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # <-- thay token của bạn

# Tạo tunnel
public_url = ngrok.connect(8000)
print(f"🌐 Frontend URL: {public_url}")
print(f"📡 API URL: {public_url}/docs")
```

```python
# Chạy server (cell này sẽ block)
!python server.py
```

> Mở `public_url` trong browser → dùng UI như bình thường (upload ảnh, click mask, complete).

---

## Xử lý lỗi thường gặp

### ❌ CUDA Out of Memory

```python
# Giải pháp 1: Clear cache
import torch
torch.cuda.empty_cache()

# Giải pháp 2: Giảm SAM grid density
masks = segmenter.segment_everything(img, points_per_side=16)  # giảm từ 32 → 16

# Giải pháp 3: Restart runtime
# Runtime → Restart runtime → chạy lại từ đầu
```

### ❌ Module not found

```python
# Đảm bảo đang ở đúng thư mục
import os
print(os.getcwd())  # Phải là /content/AP_ProtoSAM_Amodal

# Nếu sai, chạy:
%cd /content/AP_ProtoSAM_Amodal
```

### ❌ SAM checkpoint not found

```python
# Download lại
!wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### ❌ Cập nhật code mới từ GitHub

```python
# KHÔNG cần clone lại, chỉ cần pull
!git pull
```

---

## Bảng VRAM tham khảo

| Bước | VRAM sử dụng | Ghi chú |
|------|:---:|---------|
| SAM2 segment | ~6 GB | Giải phóng sau khi segment xong |
| SD2 inpainting | ~5 GB | Đã bật attention slicing |
| CLIP | ~0.5 GB | Chạy song song với SD2 |
| **Peak (có optimize)** | **~6 GB** | Nhờ load tuần tự SAM → SD2 |
| **Peak (không optimize)** | **~14 GB** | Nếu giữ cả SAM + SD2 trong VRAM |

> 💡 **Tip**: Luôn `del segmenter` + `torch.cuda.empty_cache()` sau bước segmentation để giải phóng ~6GB trước khi load SD2.

---

## Tối ưu cho từng loại GPU

| GPU | Khuyến nghị |
|-----|-------------|
| **T4 (16GB)** | Load tuần tự (SAM → free → SD2). `points_per_side=32`, `max_iter=3` |
| **A100 (40GB)** | Có thể giữ cả 2 model. Tăng `points_per_side=48`, `max_iter=4` cho chất lượng cao hơn |
| **V100 (16GB)** | Giống T4. Nhanh hơn T4 ~20% nhờ FP16 Tensor Cores |
