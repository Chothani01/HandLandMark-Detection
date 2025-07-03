# ✋🤚 Real-time Hand Tracking using MediaPipe and OpenCV

This project demonstrates **real-time hand tracking and landmark detection** using [MediaPipe](https://google.github.io/mediapipe/) and OpenCV in Python.  
It includes two scripts: a **simple direct example**, and a **class-based modular version** with extra functionality like finger state detection.

---

## 📂 Files

### ✅ `HandTrackingModule.py` (Class-based)

- Contains `handDetector` class.
- Features:
  - Detect and draw hand landmarks and connections.
  - Get all landmark positions as `[id, x, y]`.
  - Check which fingers are up (e.g., for gesture recognition).

- Main methods:
  - `findHands(img, draw=True)`: Detects hands and draws landmarks.
  - `findPosition(img, handNo=0, draw=True)`: Returns landmark positions of a specified hand.
  - `fingerUps()`: Returns list showing which fingers are up (1) or down (0).

---

### ✅ `HandTrackingBasic.py` (Simple example)

- Directly uses MediaPipe `Hands` solution.
- Draws landmarks and connections on one hand (default).
- Prints landmark coordinates in console.
- Calculates and displays FPS on webcam feed.

---

## 🚀 Features

- Real-time detection of **21 hand landmarks**.
- FPS display on the frame.
- Optional finger up/down detection logic in class version.
- Modular and easy to integrate in other projects (e.g., gesture control).

---

## ⚙️ Installation

```bash
pip install opencv-python mediapipe numpy
