from ultralytics import YOLO
paths = [
    r'models/room_best.pt',
    r'models/best1.pt',
    r'models/yolo11n.pt',
    r'models/yolo11s.pt',
]
for p in paths:
    try:
        m = YOLO(p)
        names = getattr(m, 'names', {})
        print(f'[OK] {p} -> {len(names)} classes')
    except Exception as e:
        print(f'[FAIL] {p}: {e}')
