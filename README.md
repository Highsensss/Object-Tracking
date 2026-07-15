# 🎯 Live Object Tracker

Real-time object detection and tracking with a live magnified zoom overlay — built on **YOLOv8** and **ByteTrack**.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-00FFFF.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-powered-5C3EE8.svg)

It detects objects from a live camera feed, locks onto the largest one, and shows a magnified zoom-in of it in the corner of the screen while tracking stays locked to that object across frames.

```bash
pip install -r requirements.txt
python main.py
```

Press `Esc` to quit.
