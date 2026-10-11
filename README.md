# Gesture-Controlled Air Mouse

A computer vision project that lets you control your computer's mouse pointer and perform clicks using hand gestures. Built with Python, OpenCV, MediaPipe, NumPy, and AutoPy.

## Features

- Real-time hand tracking through a webcam
- Mouse movement using the index fingertip
- Smooth cursor movement
- Gesture-based mouse clicking
- On-screen gesture status and FPS display
- Configurable camera dimensions and cursor smoothing

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- AutoPy

## How It Works

1. The webcam captures video frames.
2. MediaPipe detects hand landmarks.
3. The index fingertip controls pointer movement when the index finger is raised and the middle finger is lowered.
4. Raising both the index and middle fingers activates clicking mode.
5. Bringing their fingertips within a short distance triggers a mouse click.
6. Cursor smoothing reduces abrupt movements.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the module:

```bash
python AirMouseModule.py
```

Allow webcam access and keep your hand inside the highlighted region.

- **Move:** Raise your index finger while keeping your middle finger lowered.
- **Click:** Raise both fingers and bring their fingertips close together.
- **Exit:** Press `q` in the webcam window.

## Project Structure

```text
Gesture-Air-Mouse/
├── AirMouseModule.py
├── README.md
└── requirements.txt
```

## Requirements and Limitations

- A working webcam is required.
- Mouse-control compatibility depends on the operating system and AutoPy.
- Cursor movement and gesture recognition may need calibration for different camera setups.
- This project controls the actual system mouse, so close important windows and save your work before testing.
- The project should be tested on the target computer before use.

## Author

Developed as a computer vision and gesture-control learning project.
