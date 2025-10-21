from ultralytics import YOLO
import os, glob

MODEL_PATH = r'models/best1.pt'
LABELS_DIR = r'C:\Users\aruls\Documents\yolo_room_training_starter\dataset\labels\train'

model = YOLO(MODEL_PATH)
names = getattr(getattr(model,'model',None),'names',{})
if isinstance(names, list):
    names = {i:n for i,n in enumerate(names)}

all_classes = set()
for txt in glob.glob(os.path.join(LABELS_DIR, '*.txt')):
    with open(txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                all_classes.add(int(parts[0]))
            except:
                pass

missing = [i for i in sorted(all_classes) if i not in names]
matched = [names.get(i, '?') for i in sorted(all_classes) if i in names]

print('labels_present:', sorted(all_classes))
print('model_names:', names)
print('matched_label_names:', matched)
print('missing_indices:', missing)
print('SUMMARY:', 'OK (model covers all label indices)' if not missing else f'MISSING {missing}')
