import time, gc, os, shutil, pathlib
from ultralytics import YOLO

outdir = os.getcwd()
name = 'yolo11s.pt'
print(f'-> Preparing {name} ...')

m = YOLO(name)  # triggers auto-download to cache
ckpt = pathlib.Path(getattr(m, 'ckpt_path', ''))
if not ckpt.is_file():
    raise FileNotFoundError(f'Could not locate cached file for {name} at {ckpt}')

dst = pathlib.Path(outdir) / name

# release references & give Windows a moment to unlock the file
del m
gc.collect()
time.sleep(0.5)

for i in range(6):
    try:
        shutil.copy2(ckpt, dst)
        print(f'   Saved: {dst}')
        break
    except PermissionError:
        print(f'   Locked (attempt {i+1}/6). Retrying...')
        time.sleep(0.8)
else:
    raise SystemExit("Failed to copy after retries.")
