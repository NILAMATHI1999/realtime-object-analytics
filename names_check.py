from ultralytics import YOLO
import json

for p in [r'models/room_best.pt', r'models/best1.pt']:
    try:
        m = YOLO(p)
        names = getattr(getattr(m,'model',None),'names',{})
        if isinstance(names, list):
            names = {i:n for i,n in enumerate(names)}
        print(p, '->', json.dumps(names, ensure_ascii=False))
    except Exception as e:
        print(p, '-> ERROR:', e)
