# Multimodal Driver Drowsiness Detection System 😴🚗🛡️

A modern, real-time safety monitoring application designed to detect driver fatigue, yawning, and distraction. The project utilizes a hybrid detection approach combining high-performance client-side computer vision (via MediaPipe Face Mesh in the browser) and deep learning sequential models in the backend (using a PyTorch CNN and Spatio-Temporal Graph Neural Networks).

---

## 🌟 Key Features

* **📷 Live Webcam Monitoring**: Real-time video processing inside the browser using MediaPipe Face Mesh for low-latency facial landmark extraction.
* **🧠 Deep Learning Classifier**: Sequential evaluation of the last 30 frames using a CNN combined with Spatio-Temporal Graph Neural Networks (ST-GCN) in the Flask backend.
* **⚠️ Tri-state Safety Warnings**:
  * **Drowsy Alert**: Triggered by prolonged eye closure (Eye Aspect Ratio (EAR) < 0.25).
  * **Yawn Alert**: Triggered by yawning (Mouth Aspect Ratio (MAR) > 0.60).
  * **Distraction Alert**: Triggered by looking away from the road (yaw head orientation tolerance > 0.15).
* **📁 Static & Video File Analyzer**: Drop photos or video clips into the upload zone to perform offline classification.
* **🎛️ Dynamic Calibration**: Adjust tracking sensitivity and aspect ratio thresholds in real-time to fit different lighting and facial geometries.
* **🗄️ Database Session Logs**: Saves driving logs (duration, blinks, yawns, peak fatigue, safety status) locally inside an SQLite database.
* **📱 Responsive Design**: A layout that fits desktop monitors and collapses into a mobile-friendly slide-out drawer on phones.

---

## 🛠️ Architecture Stack

* **Frontend**: React.js, Vite, Tailwind CSS, Framer Motion (animations), Recharts (live charts).
* **AI Model (Client-side)**: MediaPipe FaceMesh (CDN WASM import).
* **Backend API**: Python, Flask, PyTorch (Deep Learning inference), SQLite (Data logging).

---

## 🚀 Installation and Run Guide

### 1. Backend Setup (Python)
Navigate to the `backend` folder and follow these steps:

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision flask flask-cors pillow
   ```
3. Run the Flask server:
   ```bash
   python server.py
   ```
   *The server runs locally at `http://127.0.0.1:5000`.*

### 2. Frontend Setup (React)
Navigate to the `frontend` folder and follow these steps:

1. Install Node modules:
   ```bash
   npm install
   ```
2. Start the local development server:
   ```bash
   npm run dev
   ```
   *Open `http://localhost:5173` in your browser to view the application.*

---

## 📂 Project Structure

```text
├── backend/
│   ├── database.py       # SQLite database configuration & schema
│   ├── server.py         # Flask API endpoint for Deep Learning inference
│   └── drownshield.db    # SQLite local logs file
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   └── Dashboard.jsx   # Main application dashboard page
│   │   ├── components/
│   │   │   └── AudioAlarm.js   # Web Audio alert oscillator alarm module
│   │   ├── App.jsx
│   │   └── index.css           # Styling styles and responsive variables
│   └── package.json
└── models/
    ├── deep_learning_models.py # PyTorch CNN & ST-GCN network layers
    └── drowsiness_model.pth    # Serialized model weight parameters
```

---

## 🛡️ License

This project is licensed under the MIT License.
