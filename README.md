# ASL Landmark Recognition (MediaPipe + TensorFlow)

Overview
- Small project to collect hand landmark vectors (MediaPipe), train a simple classifier, and run real‑time ASL letter recognition.
- Data is stored as .npy landmark vectors in `landmark_dataset/<LABEL>/` (e.g. A..Z, space, fullstop).
- Trained model and labels are saved under `landmark_model/`.
- Realtime scripts:
  - `data.py` — capture landmarks and save .npy files into labeled folders.
  - `train.py` — load dataset, train a dense NN, save model.
  - `test.py` — run realtime inference and overlay predicted label.
  - `test_sentence.py` — build sentences from predictions and publish via MQTT.
  - `speak.py` — MQTT subscriber to speak received sentences (uses espeak).

Files
- data.py: capture script using OpenCV + MediaPipe. Saves 63‑dim vectors (21 landmarks × 3).
- train.py: builds/trains a small MLP (input shape (63,)) and saves model + label file.
- test.py: realtime single‑letter prediction display.
- test_sentence.py: aggregates predictions into sentences, publishes final sentences to MQTT topic.
- speak.py: subscribes to MQTT, writes `received_text.txt`, and uses espeak to speak.
- landmark_dataset/: captured .npy files (per‑class folders).
- landmark_model/: saved Keras model and labels (not tracked by default — see .gitignore below).

Requirements
- Python 3.8+
- Packages (example install):
  - pip install opencv-python mediapipe numpy tensorflow scikit-learn paho-mqtt

On Raspberry Pi / Linux also install:
- espeak (or your TTS) for speak.py: sudo apt install espeak

Quick setup (Windows PowerShell)
- Create venv and install:
  ```
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  pip install -r requirements.txt    # or install packages listed above
  ```
- (Optional) Install espeak on Pi if using speak.py.

How to collect data
1. Connect webcam and run:
   ```
   python data.py
   ```
2. Follow on‑screen label display. Press keys (a..z, space, etc.) to change `current_label`.
   - Each detected hand frame saves a timestamped .npy vector into `landmark_dataset/<label>/`.
3. Stop capture with 'q' (or close window).

Notes: data.py creates `landmark_dataset` automatically. Verify saved samples with a quick Python check:
```
import numpy as np, glob
path = r"landmark_dataset\A\*.npy"
f = glob.glob(path)[0]
print(np.load(f).shape)
```

Train the model
```
python train.py
```
- Trains for 20 epochs (default in script) and saves to `landmark_model/landmark_model.h5` and `labels.txt`.
- Ensure you have enough samples per class and no empty folders.

Realtime inference
- Single letter:
  ```
  python test.py
  ```
- Sentence mode (aggregates letters; publishes via MQTT):
  - Edit MQTT broker IP in `test_sentence.py` (variable `broker_address`) to your broker.
  ```
  python test_sentence.py
  ```
- Speaker (subscribe + TTS):
  - Edit broker IP in `speak.py` and run:
  ```
  python speak.py
  ```
  - When a final sentence (contains '.') arrives, it will prompt for language code and invoke espeak.

Troubleshooting & tips
- Camera index: if no feed, change cv2.VideoCapture(0) index.
- Model mismatch: ensure model input shape is 63 and landmark order matches capture.
- If training fails due to class imbalance, add more samples or remove low-sample classes.
- On Windows, espeak usage differs; prefer running speak.py on Pi/Linux.

