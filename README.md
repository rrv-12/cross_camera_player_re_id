# Cross-Camera Player Tracking and Re-Identification

This project implements a **cross-camera player re-identification system** for sports videos using deep learning-based detection, feature extraction, attention mechanisms, and the Hungarian algorithm for identity matching.

It processes input from multiple camera feeds, detects players using a YOLOv8 model, extracts visual features using ResNet and a custom COSAM attention module, and re-identifies players across views using feature similarity.

---

## Key Features

- **YOLOv8-based Player Detection** for robust object localization.
- **Custom COSAM (Channel-wise Self Attention Module)** for enhanced feature discrimination.
- **Re-Identification across cameras** using cosine similarity and the **Hungarian algorithm** for optimal identity assignment.
- **EasyOCR integration** for jersey number recognition.
- **Logging support** for debugging and result tracking.

---

## Project Structure

- `cross_camera_player_mapping.ipynb` – Main notebook with end-to-end pipeline: detection, feature extraction, identity mapping.
- Uses custom models and built-in libraries (`YOLO`, `ResNet`, `COSAM`, `Hungarian matching`, `EasyOCR`).
- Logging to both console and file (`player_tracking_debug.log`).

---

## Installation

Install the required packages:

```bash
pip install ultralytics torch torchvision opencv-python faiss-cpu numpy scipy easyocr
```

---

## Usage

1. Clone the repository and open the notebook.
2. Run all cells sequentially:
   - It initializes models (YOLOv11, ResNet50).
   - Runs detection on frames from different cameras.
   - Extracts features and performs re-identification.
   - Logs and optionally visualizes player mapping results.

3. Example outputs include:
   - Player ID assignments across multiple views.
   - OCR-based jersey number overlays.
   - Re-identification accuracy logs.

---

## Core Modules

### COSAM Module (Channel-wise Self-Attention Mechanism)

Used to improve feature representation from convolutional layers:

```python
class COSAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        ...
```

### Matching Algorithm

Re-identification is done by computing cosine distances between player embeddings and minimizing total distance using:

```python
from scipy.optimize import linear_sum_assignment
```

---

## Applications

- Sports analytics and highlights.
- Player performance tracking.
- Automated multi-view surveillance systems.

---

## Notes

- Requires GPU for real-time performance.
- Pre-trained weights are needed for YOLOv11 and ResNet50.
- Ensure consistent frame rates across camera feeds.
