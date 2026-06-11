# DrowsiShield: Multimodal Driver Drowsiness Detection System 😴🚗🛡️

DrowsiShield is a real-time safety application designed to detect driver fatigue, yawning, and distractions. The project utilizes a hybrid detection approach combining high-performance client-side computer vision (via MediaPipe Face Mesh in the browser) and deep learning sequential models in the backend (using a PyTorch CNN and Spatio-Temporal Graph Neural Networks).

---

## 🌟 Key Features

* **📷 Live Webcam Monitoring**: Real-time video processing inside the browser using MediaPipe Face Mesh for low-latency facial landmark extraction.
* **🧠 Deep Learning Classifier**: Sequential evaluation of the last 30 frames using a CNN combined with Spatio-Temporal Graph Neural Networks (ST-GCN) in the Flask backend.
* **⚠️ Tri-state Safety Warnings**:
  * **Drowsy Alert**: Triggered by prolonged eye closure (Eye Aspect Ratio (EAR) < 0.25).
  * **Yawn Alert**: Triggered by yawning (Mouth Aspect Ratio (MAR) > 0.60).
  * **Distraction Alert**: Triggered by looking away from the road (yaw head orientation tolerance > 0.15).
* **📱 Web Audio Oscillator Alarm**: Microsecond-latency alarms built directly into the browser (alert sounds play instantly and dynamically adjust pitch based on threat level).
* **🗄️ Database Session Logs**: Saves driving logs (duration, blinks, yawns, peak fatigue, safety status) locally inside an SQLite database.
* **📱 Fully Responsive Design**: Built with Tailwind CSS and Framer Motion, optimized for desktop screens and collapsible to mobile viewports.

---

## 🛠️ Architecture Stack

* **Frontend**: React.js, Vite, Tailwind CSS, Framer Motion, Recharts (live charts).
* **Client-Side AI**: MediaPipe FaceMesh (WASM browser execution).
* **Backend API**: Python, Flask, PyTorch (Deep Learning inference), SQLite (database logging).
* **Routing**: Vercel Server-side proxy rewrites (routing `/api/*` to Render automatically).

---

## 🚀 Deployment Guide (Cloud)

The application is fully configured to run completely in the cloud with automated deployments:

### 1. Frontend Deployment (Vercel)
The frontend is hosted on Vercel and is configured to automatically proxy backend API requests using `vercel.json` (no CORS errors and no manual URL pasting).
* **Live Link**: [https://frontend-ebon-alpha-45.vercel.app](https://frontend-ebon-alpha-45.vercel.app)
* **To redeploy**: Push any commit to the `main` branch, and Vercel will rebuild automatically.

### 2. Backend Deployment (Render)
The backend is hosted as a Python web service on Render, managed using the `render.yaml` Blueprint definition file.
* **Live Link**: [https://driver-drowsiness-detection-v4p7.onrender.com](https://driver-drowsiness-detection-v4p7.onrender.com)
* **To redeploy**: Go to Render Blueprints dashboard, connect the repository, and click Apply.

---

## 💻 Local Quick Start Guide

If you want to run the project locally on your machine:

### Method A: One-Click Launcher (Windows)
Double-click the **`deploy_local.bat`** file at the root of the project. The script will:
1. Verify if the frontend needs to be built.
2. Search for and activate any local Python virtual environments.
3. Start the Flask application server on `http://localhost:5000`.

### Method B: Docker Containerization
If you have Docker installed on your machine, you can run the entire multi-stage build in one command:
```bash
docker-compose up --build -d
```
This will containerize both the compiled React frontend and the PyTorch backend, exposing the application on port `5000`.

### Method C: Manual Setup

#### 1. Backend Setup (Python)
Navigate to the `backend` folder:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r ../requirements.txt
python server.py
```

#### 2. Frontend Setup (React)
Navigate to the `frontend` folder:
```bash
npm install
npm run dev
```

---

## 📂 Project Structure

```text
├── Dockerfile               # Multi-stage Docker config (builds React + runs PyTorch)
├── docker-compose.yml       # Orchestrates app containers and database volumes
├── render.yaml              # Render infrastructure blueprint configuration
├── deploy_local.bat         # One-click Windows local production launcher
├── requirements.txt         # Backend Python dependencies
├── backend/
│   ├── database.py          # SQLite database schema configurations
│   ├── server.py            # Flask API (Evaluation, Session Log, Predict endpoints)
│   └── drowsishield.db      # SQLite local database logs
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   └── Dashboard.jsx # Main Dashboard UI and tracking page
│   │   ├── components/
│   │   │   └── AudioAlarm.js # Web Audio alert module
│   │   ├── App.jsx
│   │   └── index.css         # Styling system
│   ├── vercel.json          # Vercel reverse proxy rewrite configs
│   ├── vite.config.js       # Vite build configs
│   └── package.json
└── models/
    ├── deep_learning_models.py # PyTorch CNN & ST-GCN network layers
    └── drowsiness_model.pth    # Deep learning weights parameters file
```

---

## 🛡️ License

This project is licensed under the MIT License.
