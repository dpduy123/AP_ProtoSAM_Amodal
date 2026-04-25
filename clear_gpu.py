import torch
import gc

def force_clear_gpu():
    """Hàm dọn dẹp GPU thủ công cực mạnh"""
    # Xóa các biến global nếu có
    for var in ['completer', 'evaluator', 'model', 'pipe', 'processor']:
        if var in globals():
            print(f"--- Đang xóa biến: {var} ---")
            del globals()[var]
            
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU đã được giải phóng hoàn toàn!")

# Gọi hàm
force_clear_gpu()
