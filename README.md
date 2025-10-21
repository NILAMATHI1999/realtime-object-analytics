# 🧠 Realtime Object Analytics (YOLOv8 + Streamlit)

> Interactive web app for **real-time detection and analysis** of common indoor objects.  
> **Detected Classes:** person, chair, cell phone, cup, bottle, couch, dining table  

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10–3.13-blue" />
  <img src="https://img.shields.io/badge/ultralytics-8.3.217-orange" />
  <img src="https://img.shields.io/badge/streamlit-app-red" />
</p>

---

## 📋 Full Guide (Quickstart + Structure + Features + More)

```bash
# 🚀 QUICKSTART

# 1️⃣ Create and activate environment
python -m venv .venv
.\.venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the Streamlit app
streamlit run src/app_streamlit.py --server.port 8501


# 📁 PROJECT STRUCTURE

models/
 ├── best_refit.pt          # Main trained YOLOv8 weights
 └── best1.pt               # Earlier version (optional)

src/
 └── app_streamlit.py       # Main Streamlit interface

test_images/                # Sample test images
videos/                     # Optional demo clips
room_train_as_val.yaml      # Dataset reference (labels/classes)
requirements.txt


# 🧩 FEATURES

✅ Real-time object detection (YOLOv8)
✅ Streamlit web dashboard for image/video uploads and webcam mode
✅ Automatic snapshot saving and CSV logging
✅ Lightweight analytics with per-class summary
✅ Adaptive brightness normalization and confidence fallback
✅ Optimized sidebar for class filtering and performance modes


# 📈 MODEL INFO

• Fine-tuned YOLOv8 model: best_refit.pt  
• Trained on a custom indoor dataset (7 object classes)  
• Balanced for real-world room environments  
• Ideal for robotics, HRI, and scene-understanding applications


# 📸 DEMO PREVIEW

(Add your Streamlit screenshots or a short demo video here later)
Example:
![App Demo](docs/demo_screenshot.png)


# 🧰 TECH STACK

• Python 3.10–3.13  
• Ultralytics YOLOv8 (v8.3.217)  
• Streamlit for UI  
• OpenCV, Pillow, Pandas, Matplotlib


# 🌟 FUTURE ADD-ONS

• Performance presets (Speed / Balanced / Quality)  
• Hybrid snapshots (contact sheet + configurable frames)  
• Frame pacing control (e.g., 20 ms delay option)  
• Grad-CAM visualization for model explainability  
• FPS benchmark mode and ONNX/quantized export  
• Domain adaptations (medical / robotics / gesture datasets)  
• Auto-generated analytics report (PDF + charts)
