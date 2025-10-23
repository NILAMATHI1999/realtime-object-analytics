from ultralytics import YOLO
import torch, platform, json

print("Torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| Python:", platform.python_version())

models = ["models/room_best.pt", "models/best1.pt"]
for p in models:
    print("\n=== ", p, " ===", sep="")
    try:
        m = YOLO(p)
        names = getattr(m, "names", {})
        ncls = len(names) if isinstance(names, (list, dict)) else "n/a"
        stride = getattr(getattr(m, "model", None), "stride", None)
        stride_val = stride.tolist() if hasattr(stride, "tolist") else stride
        print("num_classes:", ncls)
        print("names:", names)
        print("stride:", stride_val)
    except Exception as e:
        print("ERROR:", e)
