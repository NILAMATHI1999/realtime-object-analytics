from ultralytics import YOLO
import pathlib, shutil, os

outdir = os.getcwd()  # current folder = models
os.makedirs(outdir, exist_ok=True)

for name in ['yolo11n.pt', 'yolo11s.pt']:
    print(f'-> Preparing {name} ...')
    m = YOLO(name)              # triggers auto-download to Ultralytics cache
    ckpt = pathlib.Path(getattr(m, 'ckpt_path', ''))
    if not ckpt.is_file():
        raise FileNotFoundError(f'Could not locate cached file for {name}.')
    dst = pathlib.Path(outdir) / name
    shutil.copy2(ckpt, dst)
    print(f'   Saved: {dst}')
print('Done.')
