# GestureHID: Real-Time Vision-Based Gesture-Controlled HID System

GestureHID is a real-time gesture recognition system that translates hand gestures into USB Human Interface Device (HID) inputs, such as mouse movements and keyboard commands. Designed to run efficiently on the NVIDIA Jetson Nano, this system utilizes MediaPipe Hands for real-time hand tracking and a custom-trained gesture classification model to enable touch-free, intuitive control of digital systems.

This project was developed as part of an advanced research initiative in real-time computer vision and human-computer interaction.

---

## Features

- **MediaPipe Hands Integration** – Real-time hand landmark tracking from a video stream
- **Custom MLP Classifier** – Classifies static gestures with high accuracy
- **Finite State Machine (FSM)** – Ensures stability and noise-resilience for gesture transitions
- **USB HID Emulation** – Converts recognized gestures into keyboard/mouse inputs using Linux USB Gadget API
- **Portable & Efficient** – Runs on a Jetson Nano with GPU acceleration

---

## Project Structure

```bash
GestureHID/
├── dev/                       # System development and testing files
│   └── scripts/
│       └── DatasetCreation.py # Tool for creating gesture datasets
├── models/                    # Pretrained models for gesture classification
├── src/                       # Main system modules
│   ├── GestureDemo.py         # Live gesture classification demo
│   ├── GestureFSM.py          # FSM for robust gesture state transitions
│   ├── GestureHID.py          # Main execution script with HID output
│   ├── HIDController.py       # Interfaces with USB Gadget HID API
│   └── Model.py               # PyTorch model definition
├── requirements.txt
└── README.md
```

---

## Running the Demos

The system’s behavior can be visualized and tested using the demos in the `src/` directory.

### 1. Install Python Dependencies

From the project root directory:

```bash
pip install -r requirements.txt
```

### 2. Run Gesture Classification Demo

Launch the live classification demo with real-time prediction:

```bash
cd src/
python GestureDemo.py
```

### 3. Label Custom Gestures for Training

To collect and label your own gestures:

```bash
cd dev/scripts/
python DatasetCreation.py
```

You will be prompted to:
- Enter a name for your gesture
- Specify the number of samples (default is 500)
- Use the webcam to record the gesture (press `s` to start recording, `q` to stop)

Captured data will be saved to `data/dataset_new.csv` in CSV format with rows structured as:
```text
gesture_name, x1, y1, z1, x2, y2, z2, ..., x21, y21, z21
```

Each row represents one hand pose using the 3D landmarks provided by MediaPipe.

This dataset can then be used to retrain or expand the gesture classification model from the `dev/src/` directory.

---

## Running on NVIDIA Jetson Nano

Before running the HID system on Jetson Nano, ensure the following setup steps are completed:

### Hardware & OS

- NVIDIA Jetson Nano running **JetPack 4.6 or later**
- Python 3.8 or later installed

### System Configuration

- USB Gadget mode enabled (`usb_f_hid`)
- Proper `configfs` setup or usage of `g_hid` for HID emulation
- Access permissions granted for:
  ```bash
  /dev/hidg0
  /dev/hidg1
  ```
  Root access is typically required to write to these files:
  ```bash
  sudo python3 src/GestureHID.py
  ```

### Optional (Recommended)

- Add `udev` rules for persistent access to `/dev/hidg*`
- Use a virtual environment for dependency isolation

---

## License

This project is licensed under the MIT License.

---

## Author

**Daniel Gebura**  
Graduate Student, Rochester Institute of Technology  
djg2170@rit.edu

For inquiries or contributions, feel free to contact or submit a pull request.