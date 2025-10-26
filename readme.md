# YOLO_OpenCV — Object Detection Module

A simple object detection module using Ultralytics YOLO and OpenCV.

## Files
- `ObjectDetectionModule.py` — detection class and camera demo.
- `best.pt` — default model weights (expected in repo root)/ You can also Use your own models .
- `Requirements.txt` — Python dependencies.

## Setup (Windows)
1. Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```
2. Install dependencies:
```powershell
pip install -r requirements.txt
```
Note: For CUDA support, install a matching `torch` wheel from https://pytorch.org/ instead of the generic `pip install` above.

## Run
Start the camera demo:
```powershell
python ObjectDetectionModule.py
```
Press `q` to quit the demo window.

## Usage notes
- To use a different model, update the `model_path` in the `__main__` section or instantiate `ObjectDetectionModule(model_path='path/to/model.pt')`.
- If running on a headless server, install `opencv-python-headless` and avoid GUI functions (`cv2.imshow` / `cv2.waitKey`).
- If you see issues with `ultralytics` expecting a specific `torch` variant, follow the official PyTorch install instructions for your OS/GPU.