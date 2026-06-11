import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, Sliders, Database, Video, UploadCloud, 
  Volume2, VolumeX, AlertOctagon, TrendingUp, RefreshCcw, 
  Terminal, BarChart2, Shield, Clock, CheckCircle, Eye, Zap,
  Menu, X
} from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { startAlarm, stopAlarm, triggerQuickBeep, setMuted, initAudio } from '../components/AudioAlarm';

// Dynamic import of MediaPipe from CDN to prevent Vite bundling conflicts with WebAssembly
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs";

// ── Facial Mesh Landmark Indices ──────────────────────────────────────────────
const LEFT_EYE  = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE = [33,  160, 158, 133, 153, 144];
const MOUTH     = [61,  82,  312, 291, 317, 87];
const NOSE_TIP    = 1;
const LEFT_CHEEK  = 234;
const RIGHT_CHEEK = 454;

// ── Drowsiness Detection Thresholds ───────────────────────────────────────────
const CONSEC_FRAMES_EAR = 15;  // ~15 frames of closed eyes = drowsy (~0.5s at 30fps)
const CONSEC_FRAMES_MAR = 12;  // ~12 frames open mouth = yawning
const CONSEC_FRAMES_YAW = 18;  // ~18 frames head turned = distracted

function Dashboard() {
  const userId = 1;

  // ── Tab & Input Mode ────────────────────────────────────────────────────────
  const [activeTab, setActiveTab]   = useState('dashboard');
  const [inputMode, setInputMode]   = useState('webcam');
  const [backendUrl, setBackendUrl] = useState(() => {
    const saved = localStorage.getItem('drowsishield_backend_url');
    return saved ? saved.replace(/\/$/, '') : window.location.origin.replace(/\/$/, '');
  });
  const [uploadedImageName, setUploadedImageName] = useState(null);
  const [staticImage, setStaticImage] = useState(null);
  const [staticLandmarks, setStaticLandmarks] = useState(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
  // ── Detection Threshold Settings ────────────────────────────────────────────
  const [earThreshold, setEarThreshold] = useState(0.25);
  const [marThreshold, setMarThreshold] = useState(0.60);
  const [yawTolerance, setYawTolerance] = useState(0.15);
  const [isAudioMuted, setIsAudioMuted] = useState(false);

  // ── Real-time Detection Metrics ─────────────────────────────────────────────
  const [ear,             setEar]             = useState(0.0);
  const [mar,             setMar]             = useState(0.0);
  const [blinkRate,       setBlinkRate]       = useState(0.0);
  const [blinks,          setBlinks]          = useState(0);
  const [yawns,           setYawns]           = useState(0);
  const [closureDuration, setClosureDuration] = useState(0.0);
  const [headDir,         setHeadDir]         = useState('Forward');
  const [emotion,         setEmotion]         = useState('😐 Neutral');
  const [presence,        setPresence]        = useState('👤 Alone');
  const [fatigue,         setFatigue]         = useState(0.0);
  const [fps,             setFps]             = useState(0);
  const [driverState,     setDriverState]     = useState('ACTIVE');
  const [dlPrediction,    setDlPrediction]    = useState('ALERT');
  const [dlProbability,   setDlProbability]   = useState(0.0);
  const [landmarkCount,   setLandmarkCount]   = useState(0);

  // ── AI Evaluation Metrics ───────────────────────────────────────────────────
  const [evalMetrics, setEvalMetrics] = useState({
    accuracy: 0.95, precision: 0.941, recall: 0.96, f1_score: 0.95, dataset_size: 200
  });

  // ── UI State ─────────────────────────────────────────────────────────────────
  const [loading,         setLoading]         = useState(true);
  const [logs,            setLogs]            = useState([]);
  const [sessionHistory,  setSessionHistory]  = useState([]);
  const [chartData,       setChartData]       = useState([]);
  const [webcamError,     setWebcamError]     = useState(null);
  const [isCameraActive,  setIsCameraActive]  = useState(false);

  // ── Session Tracking ──────────────────────────────────────────────────────── 
  const [sessionStart,        setSessionStart]        = useState(Date.now());
  const [maxFatigueRecorded,  setMaxFatigueRecorded]  = useState(0.0);
  const [sessionDuration,     setSessionDuration]     = useState(0);

  // ── Refs ──────────────────────────────────────────────────────────────────── 
  const videoRef         = useRef(null);
  const canvasRef        = useRef(null);
  const fileInputRef     = useRef(null);
  const requestRef       = useRef(null);
  const faceLandmarkerRef = useRef(null);

  // Rolling tracking variables (avoid re-renders for internal state)
  const stateRef = useRef({
    earCounter:        0,
    marCounter:        0,
    yawCounter:        0,
    isBlinking:        false,
    blinkStartTime:    0,
    isYawning:         false,
    lastClosedDuration: 0,
    sessionStartTime:  Date.now(),
    lastVideoTime:     -1,
    frameCount:        0,
    fpsLastTime:       Date.now(),
    blinkTimes:        [],        // timestamps of each blink for per-minute rate
    backendCallActive: false,
    sessionId:         'session_' + Math.random().toString(36).substring(2, 11),
    stream:            null,
    lastDriverState:   'ACTIVE',  // track state transitions to avoid log spam
  });

  // ── Logging Helper ────────────────────────────────────────────────────────── 
  const addLog = (msg, type = 'info') => {
    const timeStr = new Date().toTimeString().split(' ')[0];
    setLogs(prev => [...prev.slice(-49), { time: timeStr, msg, type }]);
  };

  // ── Session Clock ─────────────────────────────────────────────────────────── 
  useEffect(() => {
    const timer = setInterval(() => {
      setSessionDuration(Math.floor((Date.now() - sessionStart) / 1000));
    }, 1000);
    return () => clearInterval(timer);
  }, [sessionStart]);

  // ── 1. Fetch Backend Evaluation Metrics ───────────────────────────────────── 
  useEffect(() => {
    fetch(`${backendUrl}/api/evaluation`)
      .then(res => res.json())
      .then(data => setEvalMetrics(data))
      .catch(() => addLog("Backend API offline. Using cached model metrics.", "warning"));

    loadHistory();
  }, [backendUrl]);

  useEffect(() => {
    addLog("Dashboard initialized. Loading MediaPipe FaceMesh model...", "info");
  }, []);

  // ── 2. Load Session History from SQLite ───────────────────────────────────── 
  const loadHistory = () => {
    fetch(`${backendUrl}/api/sessions?user_id=${userId}`)
      .then(res => res.json())
      .then(data => { if (Array.isArray(data)) setSessionHistory(data); })
      .catch(() => console.error("Failed to fetch sessions history"));
  };

  // ── 3. Save Session to Database ────────────────────────────────────────────── 
  const handleSaveSession = () => {
    const duration = Math.floor((Date.now() - sessionStart) / 1000);
    fetch(`${backendUrl}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id:      parseInt(userId),
        timestamp:    new Date().toLocaleString(),
        duration_sec: duration,
        blinks_count: blinks,
        yawns_count:  yawns,
        max_fatigue:  maxFatigueRecorded,
        status:       fatigue > 50 ? 'DROWSY_WARNING' : 'SAFE'
      })
    })
    .then(res => {
      if (res.ok) {
        addLog("Driver session successfully logged to database.", "success");
        loadHistory();
      }
    })
    .catch(() => addLog("Database server unavailable. Session log skipped.", "warning"));
  };

  // ── 4. Initialize MediaPipe Face Landmarker ─────────────────────────────────
  useEffect(() => {
    let active = true;

    async function initMediaPipe() {
      try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
        );
        const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.2,
          minFacePresenceConfidence: 0.2,
          minTrackingConfidence: 0.2
        });

        if (active) {
          faceLandmarkerRef.current = landmarker;
          addLog("✅ MediaPipe FaceMesh loaded (GPU mode).", "success");
          setLoading(false);
        }
      } catch (err) {
        addLog("GPU delegate failed. Trying CPU fallback...", "warning");
        try {
          const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
          );
          const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
              modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
              delegate: "CPU"
            },
            runningMode: "VIDEO",
            numFaces: 1,
            minFaceDetectionConfidence: 0.2,
            minFacePresenceConfidence: 0.2,
            minTrackingConfidence: 0.2
          });
          if (active) {
            faceLandmarkerRef.current = landmarker;
            addLog("✅ MediaPipe FaceMesh loaded (CPU mode).", "success");
            setLoading(false);
          }
        } catch (cpuErr) {
          addLog("❌ MediaPipe loading failed. Check network/browser settings.", "danger");
        }
      }
    }

    initMediaPipe();
    return () => { active = false; stopCamera(); };
  }, []);

  // ── Persistent Static Image Renderer ───────────────────────────────────────── 
  useEffect(() => {
    if (inputMode === 'upload' && uploadedImageName && staticImage && canvasRef.current) {
      const w = canvasRef.current.width  = 640;
      const h = canvasRef.current.height = 480;
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(staticImage, 0, 0, w, h);
      if (staticLandmarks) {
        const earVal = calculateEAR(staticLandmarks, w, h);
        const marVal = calculateMAR(staticLandmarks, w, h);
        drawMesh(ctx, staticLandmarks, earVal, marVal, false);
      }
    }
  }, [inputMode, uploadedImageName, staticImage, staticLandmarks, earThreshold, marThreshold]);



  // ── 5. Camera Controls ────────────────────────────────────────────────────── 
  const startCamera = async () => {
    stopCamera();
    setWebcamError(null);
    addLog("Requesting camera stream access...", "info");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true
      });
      stateRef.current.stream = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadeddata = () => {
          addLog("📷 Live webcam feed active.", "success");
          setIsCameraActive(true);
          if (canvasRef.current) {
            canvasRef.current.width  = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
          }
          // Force-unlock Web Audio API immediately so alarm fires instantly
          initAudio();
          // Additional unlock: play a silent buffer to unblock AudioContext
          try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const buf = ctx.createBuffer(1, 1, 22050);
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            src.start(0);
            ctx.resume();
          } catch(e) {}
          requestRef.current = requestAnimationFrame(renderLoop);
        };
      }
    } catch (err) {
      const msg = err.name === 'NotAllowedError'
        ? "Camera permission denied. Please allow camera access in browser settings."
        : err.name === 'NotFoundError'
        ? "No camera device found. Use 'File Test' mode to upload a video."
        : `Camera error: ${err.message}`;
      setWebcamError(msg);
      addLog("❌ " + msg, "danger");
    }
  };

  const stopCamera = () => {
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    if (stateRef.current.stream) {
      stateRef.current.stream.getTracks().forEach(t => t.stop());
      stateRef.current.stream = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      if (videoRef.current.src) {
        URL.revokeObjectURL(videoRef.current.src);
        videoRef.current.src = "";
      }
    }
    stopAlarm();
    setIsCameraActive(false);
  };

  // ── Camera Tab-Switching & Active State Control ──────────────────────────────────
  useEffect(() => {
    if (activeTab !== 'dashboard') {
      stopCamera();
    }
  }, [activeTab]);

  // ── 6. Metric Calculators ──────────────────────────────────────────────────── 
  const dist = (p1, p2, w, h) => Math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h);

  const calculateEAR = (lm, w, h) => {
    const eyeEAR = (pts) => {
      const A = dist(pts[1], pts[5], w, h);
      const B = dist(pts[2], pts[4], w, h);
      const C = dist(pts[0], pts[3], w, h);
      return (A + B) / (2.0 * C);
    };
    const leftE  = LEFT_EYE.map(i => lm[i]);
    const rightE = RIGHT_EYE.map(i => lm[i]);
    return (eyeEAR(leftE) + eyeEAR(rightE)) / 2.0;
  };

  const calculateMAR = (lm, w, h) => {
    const pts = MOUTH.map(i => lm[i]);
    const A = dist(pts[1], pts[5], w, h);
    const B = dist(pts[2], pts[4], w, h);
    const C = dist(pts[0], pts[3], w, h);
    return (A + B) / (2.0 * C);
  };

  const calculateYaw = (lm, w, h) => {
    const nose  = lm[NOSE_TIP].x * w;
    const left  = lm[LEFT_CHEEK].x * w;
    const right = lm[RIGHT_CHEEK].x * w;
    const span  = right - left;
    return Math.abs(span) < 0.0001 ? 0.5 : (nose - left) / span;
  };

  // earThr / marThr passed in so thresholds match user's slider settings
  const estimateEmotion = (lm, earVal, marVal, w, h, earThr, marThr) => {
    // ── Surprised: eyes wide open + mouth noticeably open ──────────────────
    if (earVal > 0.34 && marVal > marThr * 0.75) return "😲 Surprised";

    // ── Yawning: very large mouth opening ──────────────────────────────────
    if (marVal > marThr * 0.92) return "🥱 Yawning";

    // ── Sleepy: eyes clearly drooping — just above drowsy threshold ─────────
    // Range: earThr to earThr+0.05  (e.g. 0.25–0.30 with default settings)
    // Keeps Neutral reachable for normal open eyes (EAR > earThr+0.05)
    if (earVal < earThr + 0.05) return "😴 Sleepy";

    // ── Happy / Sad via mouth corner curvature ─────────────────────────────
    // MediaPipe y increases downward:
    //   Smile  → corners rise  (lower y) → curvature = lipMidY - cornerY > 0
    //   Frown  → corners drop  (higher y)→ curvature < 0
    const cornerY   = (lm[61].y + lm[291].y) / 2;
    const lipMidY   = (lm[13].y + lm[14].y)  / 2;
    const curvature = lipMidY - cornerY;

    if (curvature >  0.006) return "😊 Happy";
    if (curvature < -0.005) return "☹️ Sad";

    return "😐 Neutral";
  };

  // ── 7. Backend Deep Learning Inference ────────────────────────────────────── 
  const handleBackendInference = async (videoEl, lm, w, h) => {
    const st = stateRef.current;
    const now = Date.now();
    // Throttle backend calls to once every 600ms to prevent clogging the event loop and keep rendering smooth
    if (st.backendCallActive || (now - (st.lastBackendCallTime || 0) < 600)) return;
    st.backendCallActive = true;
    st.lastBackendCallTime = now;
    try {
      let minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
      for (const p of lm) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
      }
      const padX = (maxX - minX) * 0.15, padY = (maxY - minY) * 0.15;
      const x1 = Math.max(0, Math.floor((minX - padX) * w));
      const y1 = Math.max(0, Math.floor((minY - padY) * h));
      const x2 = Math.min(w, Math.floor((maxX + padX) * w));
      const y2 = Math.min(h, Math.floor((maxY + padY) * h));

      const tc = document.createElement("canvas");
      tc.width = tc.height = 64;
      const ctx2 = tc.getContext("2d");
      if (x2 > x1 && y2 > y1) {
        ctx2.drawImage(videoEl, x1, y1, x2 - x1, y2 - y1, 0, 0, 64, 64);
      } else {
        ctx2.drawImage(videoEl, 0, 0, w, h, 0, 0, 64, 64);
      }
      const cropBase64 = tc.toDataURL("image/jpeg", 0.7);
      const curatedLms = [362,385,387,263,373,380,33,160,158,133,153,144,61,82,312,291,317,87,1,234,454]
        .map(idx => ({ x: lm[idx].x, y: lm[idx].y, z: lm[idx].z }));

      const res = await fetch(`${backendUrl}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: st.sessionId, frame: cropBase64, landmarks: curatedLms })
      });
      if (res.ok) {
        const data = await res.json();
        setDlPrediction(data.prediction);
        setDlProbability(data.probability);
      }
    } catch { setDlPrediction("Offline"); } 
    finally { st.backendCallActive = false; }
  };

  // ── 8. Facial Mesh Drawing ────────────────────────────────────────────────── 
  const drawMesh = (ctx, lm, earVal, marVal, isMirror) => {
    const w = canvasRef.current.width;
    const h = canvasRef.current.height;
    const eyeOpen = earVal >= earThreshold;
    const mouthClosed = marVal <= marThreshold;

    const drawLoop = (indices, color, width = 1.5) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      indices.forEach((idx, i) => {
        const p = lm[idx];
        const x = isMirror ? (1.0 - p.x) * w : p.x * w;
        const y = p.y * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
    };

    // Draw landmarks with color indicators
    drawLoop(LEFT_EYE,  eyeOpen ? "#00e676" : "#ff1744", 2);
    drawLoop(RIGHT_EYE, eyeOpen ? "#00e676" : "#ff1744", 2);
    drawLoop(MOUTH, mouthClosed ? "rgba(255,255,255,0.6)" : "#ff9100", 1.5);

    // Draw landmark dots for eyes
    const eyeIndices = [...LEFT_EYE, ...RIGHT_EYE];
    eyeIndices.forEach(idx => {
      const p = lm[idx];
      const x = isMirror ? (1.0 - p.x) * w : p.x * w;
      const y = p.y * h;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = eyeOpen ? "#00e676" : "#ff1744";
      ctx.fill();
    });
  };

  // ── 9. Main Render Loop ───────────────────────────────────────────────────── 
  const renderLoop = () => {
    if (!videoRef.current || !canvasRef.current || !faceLandmarkerRef.current || inputMode !== 'webcam') return;
    const st  = stateRef.current;
    const now = Date.now();
    st.frameCount++;

    // FPS calculation
    if (now - st.fpsLastTime >= 1000) {
      setFps(st.frameCount);
      st.frameCount  = 0;
      st.fpsLastTime = now;

      // Blink rate: count blinks in the last 60 seconds
      const cutoff = now - 60000;
      st.blinkTimes = st.blinkTimes.filter(t => t > cutoff);
      setBlinkRate(st.blinkTimes.length);
    }

    const videoEl = videoRef.current;
    if (videoEl.readyState < 2 || videoEl.currentTime === st.lastVideoTime) {
      requestRef.current = requestAnimationFrame(renderLoop);
      return;
    }
    st.lastVideoTime = videoEl.currentTime;

    const w   = canvasRef.current.width;
    const h   = canvasRef.current.height;
    const ctx = canvasRef.current.getContext("2d");

    // Draw mirrored video frame
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(videoEl, 0, 0, w, h);
    ctx.restore();

    const results = faceLandmarkerRef.current.detectForVideo(videoEl, performance.now());

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const lm = results.faceLandmarks[0];
      setLandmarkCount(lm.length);
      setPresence('👤 Driver Detected');

      // ── Compute metrics ──────────────────────────────────────────────────────
      const earVal = calculateEAR(lm, w, h);
      const marVal = calculateMAR(lm, w, h);
      const yawVal = calculateYaw(lm, w, h);

      setEar(earVal);
      setMar(marVal);

      // Head direction
      const yawLow  = 0.50 - yawTolerance;
      const yawHigh = 0.50 + yawTolerance;
      let dir = "Forward";
      if (yawVal < yawLow) dir = "Left";
      else if (yawVal > yawHigh) dir = "Right";
      setHeadDir(dir);

      // Draw facial wireframe
      drawMesh(ctx, lm, earVal, marVal, true);

      // Backend DL inference (throttled)
      handleBackendInference(videoEl, lm, w, h);

      // ── Blink detection state machine ────────────────────────────────────────
      if (earVal < earThreshold) {
        if (!st.isBlinking) {
          st.isBlinking     = true;
          st.blinkStartTime = performance.now();
        }
        const dur = (performance.now() - st.blinkStartTime) / 1000.0;
        st.lastClosedDuration = dur;
        setClosureDuration(dur);
      } else {
        if (st.isBlinking) {
          st.isBlinking = false;
          st.blinkTimes.push(Date.now());
          setBlinks(prev => prev + 1);
          addLog(`👁 Blink detected: ${st.lastClosedDuration.toFixed(2)}s closure`, 'success');
        }
        setClosureDuration(0.0);
      }

      // ── Yawning detection state machine ──────────────────────────────────────
      if (marVal > marThreshold) {
        if (!st.isYawning) st.isYawning = true;
      } else {
        if (st.isYawning) {
          st.isYawning = false;
          setYawns(prev => prev + 1);
          addLog(`🥱 Yawn detected (MAR: ${marVal.toFixed(2)})`, 'warning');
        }
      }

      // Emotion — pass thresholds so detection adapts to user settings
      setEmotion(estimateEmotion(lm, earVal, marVal, w, h, earThreshold, marThreshold));

      // ── Fatigue Score ─────────────────────────────────────────────────────────
      const isLaughing = earVal < earThreshold && marVal > 0.42; // eyes squinting + mouth open = laugh
      setFatigue(prev => {
        let f = prev;
        if (earVal < earThreshold && !isLaughing) f += 0.35; // only if truly drowsy (not laughing)
        if (dir !== "Forward")     f += 0.08;
        if (marVal > marThreshold) f += 0.20;
        if (earVal >= earThreshold && dir === "Forward" && marVal <= marThreshold) f -= 0.04;
        const updated = Math.max(0.0, Math.min(100.0, f));
        setMaxFatigueRecorded(mx => Math.max(mx, updated));
        return updated;
      });

      // ── Rolling chart data ────────────────────────────────────────────────────
      setChartData(prev => {
        // Use earVal/marVal directly (not stale fatigue state) for accuracy
        const lastFatigue = prev.length > 0 ? prev[prev.length - 1].fatigue : 0;
        return [...prev.slice(-29), {
          time:    new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          fatigue: parseFloat(lastFatigue.toFixed(1)),
          ear:     parseFloat(earVal.toFixed(3)),
          mar:     parseFloat(marVal.toFixed(3))
        }];
      });

      // ── Frame counters ────────────────────────────────────────────────────────
      // Laughing signature: eyes squinting (low EAR) + mouth open (high MAR)
      // → do NOT count as drowsy; reset counter to avoid false alarm
      const laughing = earVal < earThreshold && marVal > 0.42;
      
      // If looking away, eye measurements are skewed by perspective.
      // Reset ear counter and focus on distraction (yaw) counter.
      if (dir !== "Forward") {
        st.earCounter = 0;
        st.yawCounter = st.yawCounter + 1;
      } else {
        st.earCounter = (earVal < earThreshold && !laughing) ? st.earCounter + 1 : 0;
        st.yawCounter = 0;
      }
      st.marCounter = (marVal > marThreshold) ? st.marCounter + 1 : 0;

      // ── Driver state transitions ──────────────────────────────────────────────
      let state = "ACTIVE";
      if      (st.yawCounter >= CONSEC_FRAMES_YAW) state = "LOOKING AWAY";
      else if (st.earCounter >= CONSEC_FRAMES_EAR) state = "DROWSY";
      else if (st.marCounter >= CONSEC_FRAMES_MAR) state = "YAWNING";
      setDriverState(state);

      // ── Alarm control — only act on state TRANSITIONS to avoid spam ───────────
      if (state !== st.lastDriverState) {
        if (state !== "ACTIVE") {
          // Force AudioContext unlock before starting alarm
          initAudio();
          if (state === "DROWSY") {
            startAlarm("drowsy");
            addLog(`🚨 DROWSY ALERT: Eyes closed for ${CONSEC_FRAMES_EAR}+ frames! Alarm triggered.`, 'danger');
          } else if (state === "YAWNING") {
            startAlarm("yawning");
            addLog(`⚠️ YAWN ALERT: Yawning detected — ${st.marCounter} consecutive frames.`, 'danger');
          } else {
            startAlarm("attention");
            addLog(`⚠️ DISTRACTION ALERT: Head not forward for ${CONSEC_FRAMES_YAW}+ frames.`, 'danger');
          }
        } else {
          stopAlarm();
          addLog(`✅ Driver alert resolved — state back to ACTIVE.`, 'success');
        }
        st.lastDriverState = state;
      }

    } else {
      // No face in frame
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#ff9100";
      ctx.font = "bold 18px Outfit, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("⚠ No Face Detected — Position Camera", w / 2, h / 2 - 10);
      ctx.fillStyle = "#8a99ad";
      ctx.font = "14px Outfit, sans-serif";
      ctx.fillText("Ensure your face is well-lit and visible", w / 2, h / 2 + 18);
      setLandmarkCount(0);
      setPresence('👤 No Face');
      st.earCounter = st.marCounter = st.yawCounter = 0;
      st.isBlinking = st.isYawning = false;
      // Reset lastDriverState so alarms re-trigger correctly if face returns
      if (st.lastDriverState !== 'ACTIVE') {
        st.lastDriverState = 'ACTIVE';
      }
      setDriverState('ACTIVE');
      stopAlarm();
    }

    requestRef.current = requestAnimationFrame(renderLoop);
  };

  // ── 10. File Upload Processing ────────────────────────────────────────────── 
  const processUploadedFile = (file) => {
    if (!faceLandmarkerRef.current) {
      addLog("Model not ready. Please wait...", "warning");
      return;
    }
    if (file.type.startsWith("image/")) {
      addLog(`🖼 Analyzing image: ${file.name}`, "info");
      setUploadedImageName(file.name);
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const results = faceLandmarkerRef.current.detectForVideo(img, performance.now());
          if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const lm     = results.faceLandmarks[0];
            const w      = img.naturalWidth;
            const h      = img.naturalHeight;
            const earVal = calculateEAR(lm, w, h);
            const marVal = calculateMAR(lm, w, h);
            const yawVal = calculateYaw(lm, w, h);
            
            setStaticImage(img);
            setStaticLandmarks(lm);
            setEar(earVal);
            setMar(marVal);
            setLandmarkCount(lm.length);
            setPresence('👤 Driver Detected');

            let dir = "Forward";
            if (yawVal < (0.50 - yawTolerance)) dir = "Left";
            else if (yawVal > (0.50 + yawTolerance)) dir = "Right";
            setHeadDir(dir);
            let state = "ACTIVE";
            if      (dir !== "Forward")     state = "LOOKING AWAY";
            else if (earVal < earThreshold) state = "DROWSY";
            else if (marVal > marThreshold) state = "YAWNING";
            setDriverState(state);
            if (state !== "ACTIVE") {
              triggerQuickBeep(state === "DROWSY" ? 880 : 440);
              setFatigue(prev => Math.min(100, prev + 25.0));
            } else {
              setFatigue(prev => Math.max(0, prev - 10.0));
            }
            addLog(`✅ Static analysis done: ${state} (EAR: ${earVal.toFixed(3)} [thr: ${earThreshold}], MAR: ${marVal.toFixed(3)} [thr: ${marThreshold}], Yaw: ${yawVal.toFixed(2)} [dir: ${dir}])`, "success");
          } else {
            setStaticImage(img);
            setStaticLandmarks(null);
            setLandmarkCount(0);
            setPresence('👤 Alone');
            addLog("❌ Face not found in image. Try a clearer photo.", "danger");
          }
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith("video/")) {
      addLog(`🎬 Loading video: ${file.name}`, "info");
      setUploadedImageName(null);
      stopCamera();
      if (videoRef.current) {
        videoRef.current.src = URL.createObjectURL(file);
        videoRef.current.loop  = true;
        videoRef.current.muted = true;
        videoRef.current.play().then(() => {
          addLog("▶ Video loop playback active.", "success");
          if (canvasRef.current) {
            canvasRef.current.width  = videoRef.current.videoWidth  || 640;
            canvasRef.current.height = videoRef.current.videoHeight || 480;
          }
          initAudio();
          setInputMode('webcam'); // reuse render loop
          requestRef.current = requestAnimationFrame(renderLoop);
        }).catch(err => addLog("Video playback failed: " + err.message, "danger"));
      }
    }
  };

  const handleDragOver = (e) => e.preventDefault();
  const handleDrop    = (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) processUploadedFile(f); };

  // ── 11. Mute Toggle ───────────────────────────────────────────────────────── 
  const handleToggleMute = () => {
    const muted = !isAudioMuted;
    setIsAudioMuted(muted);
    setMuted(muted);
    addLog(muted ? "🔇 Alarm muted." : "🔊 Alarm active.", "info");
  };

  // ── 12. Reset Session ─────────────────────────────────────────────────────── 
  const handleResetSession = () => {
    triggerQuickBeep(500);
    handleSaveSession();
    setBlinks(0); setYawns(0); setFatigue(0.0); setMaxFatigueRecorded(0.0);
    setSessionStart(Date.now()); setChartData([]);
    addLog("🔄 Session reset and logged.", "info");
  };

  // ── Helper: Format Duration ────────────────────────────────────────────────── 
  const fmtDur = (s) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

  // ── State-Based Style Helpers ─────────────────────────────────────────────── 
  const stateColor = driverState === 'DROWSY' ? 'text-red-400'
    : driverState !== 'ACTIVE' ? 'text-orange-400' : 'text-emerald-400';

  const stateBg = driverState === 'DROWSY'
    ? 'bg-red-500/10 border-red-500/60 shadow-[0_0_30px_rgba(255,23,68,0.25)]'
    : driverState !== 'ACTIVE'
    ? 'bg-orange-500/10 border-orange-500/50 shadow-[0_0_20px_rgba(255,145,0,0.2)]'
    : 'glass-panel border-white/8';

  const fatigueBg = fatigue > 60
    ? 'bg-gradient-to-r from-red-500 to-rose-600'
    : fatigue > 30
    ? 'bg-gradient-to-r from-orange-400 to-amber-500'
    : 'bg-gradient-to-r from-emerald-500 to-teal-400';

  const metrics = [
    { label: 'Eye Ratio', sublabel: 'EAR', value: ear.toFixed(3), danger: ear < earThreshold && ear > 0, icon: '👁' },
    { label: 'Mouth Ratio', sublabel: 'MAR', value: mar.toFixed(3), danger: mar > marThreshold, icon: '👄' },
    { label: 'Blink Count', sublabel: 'total', value: blinks, danger: false, icon: '💫' },
    { label: 'Yawns', sublabel: 'detected', value: yawns, danger: yawns > 2, icon: '🥱' },
    { label: 'Eye Closure', sublabel: 'duration', value: `${closureDuration.toFixed(2)}s`, danger: closureDuration > 0.5, icon: '⏱' },
    { label: 'Head Pose', sublabel: 'direction', value: headDir, danger: headDir !== 'Forward', icon: '🧭' },
    { label: 'Blink Rate', sublabel: 'per min', value: `${blinkRate}`, danger: blinkRate > 25 || (blinkRate < 8 && blinkRate > 0), icon: '⚡' },
    { label: 'Emotion', sublabel: 'state', value: emotion, danger: false, icon: null },
  ];

  // ─────────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <div className="flex min-h-screen bg-darkBg text-white font-outfit overflow-x-hidden relative">

      {/* Mobile Sidebar Backdrop Overlay */}
      {mobileMenuOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden transition-opacity duration-300"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className={`fixed md:static inset-y-0 left-0 z-50 w-60 glass-panel border-r border-white/[0.06] flex flex-col justify-between py-6 px-4 shrink-0 transition-transform duration-300 md:translate-x-0 bg-darkBg/95 md:bg-transparent ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}`}>

        {/* Logo */}
        <div>
          <div className="flex items-center justify-between px-2 mb-8">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center glow-blue shadow-lg">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <p className="text-[15px] font-bold tracking-tight leading-tight">DrowsiShield</p>
                <p className="text-[10px] text-blue-400 font-semibold tracking-widest uppercase">AI Safety</p>
              </div>
            </div>
            <button onClick={() => setMobileMenuOpen(false)} className="p-1.5 rounded-xl hover:bg-white/5 text-textSecondary hover:text-white md:hidden transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Nav */}
          <nav className="space-y-1 mb-6">
            {[
              { tab: 'dashboard', icon: Activity, label: 'Live Monitor' },
              { tab: 'history',   icon: Database,  label: 'Session History' },
              { tab: 'settings',  icon: Sliders,   label: 'Calibration' },
            ].map(({ tab, icon: Icon, label }) => (
              <button key={tab} onClick={() => { setActiveTab(tab); setMobileMenuOpen(false); }}
                className={`nav-btn ${activeTab === tab ? 'active' : ''}`}>
                <Icon className="w-4 h-4 shrink-0" /><span>{label}</span>
              </button>
            ))}
          </nav>

          {/* Live telemetry strip */}
          <div className="mt-2 px-1">
            <p className="text-[10px] uppercase tracking-widest text-textSecondary font-semibold mb-3 px-1">Live Telemetry</p>
            <div className="space-y-1.5">
              {[
                { k: 'EAR', v: ear.toFixed(3), bad: ear < earThreshold && ear > 0, color: 'text-blue-400' },
                { k: 'MAR', v: mar.toFixed(3), bad: mar > marThreshold, color: 'text-violet-400' },
                { k: 'Blinks/min', v: blinkRate, bad: false, color: 'text-cyan-400' },
                { k: 'Landmarks', v: landmarkCount, bad: false, color: landmarkCount > 0 ? 'text-emerald-400' : 'text-textSecondary' },
                { k: 'FPS', v: fps, bad: false, color: 'text-purple-400' },
              ].map(({ k, v, bad, color }) => (
                <div key={k} className="flex items-center justify-between px-3 py-2 rounded-xl bg-white/[0.03] border border-white/[0.05]">
                  <span className="text-[11px] text-textSecondary">{k}</span>
                  <span className={`text-[11px] font-bold font-mono ${bad ? 'text-red-400' : color}`}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Session info + reset */}
        <div className="space-y-3 px-1">
          <div className="p-3 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse inline-block"></span>
              <p className="text-[12px] font-semibold text-white">Active Session</p>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] text-textSecondary">
              <Clock className="w-3 h-3" /><span className="font-mono">{fmtDur(sessionDuration)}</span>
            </div>
          </div>
          <button onClick={() => { handleResetSession(); setMobileMenuOpen(false); }}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-orange-400/30 hover:bg-orange-400/10 text-orange-400 text-[12px] font-semibold transition-all active:scale-95">
            <RefreshCcw className="w-3.5 h-3.5" /><span>Reset &amp; Save</span>
          </button>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col overflow-y-auto">

        {/* ── Header ───────────────────────────────────────────────────────── */}
        <header className="px-5 py-4 border-b border-white/[0.06] flex justify-between items-center glass-panel sticky top-0 z-40 gap-4">
          <div className="flex items-center gap-3">
            <button onClick={() => setMobileMenuOpen(true)} className="p-2 -ml-2 rounded-xl bg-white/[0.03] border border-white/[0.08] text-textSecondary hover:text-white md:hidden hover:bg-white/5 transition-colors shrink-0">
              <Menu className="w-4 h-4" />
            </button>
            <div>
              <h1 className="text-[16px] md:text-[20px] font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-300 bg-clip-text text-transparent leading-snug">
                Multimodal Driver Drowsiness Detection
              </h1>
              <p className="text-[10px] md:text-[11px] text-textSecondary mt-0.5 flex items-center gap-2 flex-wrap">
                Real-Time Multi-Modal AI Safety System
                {landmarkCount > 0 && (
                  <span className="inline-flex items-center gap-1 text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block"></span>
                    Face Tracked · {landmarkCount} pts
                  </span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <button onClick={handleToggleMute}
              className={`w-9 h-9 md:w-10 md:h-10 rounded-xl flex items-center justify-center border transition-all ${isAudioMuted ? 'border-red-500/40 text-red-400 bg-red-500/10' : 'border-white/[0.08] text-textSecondary hover:text-white hover:bg-white/5'}`}>
              {isAudioMuted ? <VolumeX className="w-3.5 h-3.5 md:w-4 md:h-4" /> : <Volume2 className="w-3.5 h-3.5 md:w-4 md:h-4" />}
            </button>
            <button onClick={handleSaveSession}
              className="px-3 py-2 md:px-4 md:py-2 bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 text-white text-[11px] md:text-[12px] font-semibold rounded-xl transition-all flex items-center gap-1.5 active:scale-95 shadow-lg">
              <CheckCircle className="w-3.5 h-3.5" /><span className="hidden sm:inline">Log Session</span>
            </button>
          </div>
        </header>

        {/* ── Tab Content ────────────────────────────────────────────────────── */}
        <AnimatePresence mode="wait">

          {/* ════ DASHBOARD TAB ═════════════════════════════════════════════ */}
                    {activeTab === 'dashboard' && (
            <motion.div key="dashboard"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              className="p-5 grid grid-cols-1 lg:grid-cols-12 gap-5">

              {/* LEFT: Video Card */}
              <div className="lg:col-span-7 xl:col-span-8 flex flex-col">
                <div className="glass-panel rounded-2xl overflow-hidden flex flex-col" style={{ minHeight: '580px', height: '100%' }}>
                  {/* Card header */}
                  <div className="px-5 py-3 border-b border-white/[0.06] flex justify-between items-center bg-white/[0.01]">
                    <div className="flex items-center gap-2">
                      <div className="w-7 h-7 rounded-lg bg-blue-500/15 border border-blue-500/25 flex items-center justify-center">
                        <Video className="w-3.5 h-3.5 text-blue-400" />
                      </div>
                      <span className="font-semibold text-[13px]">{inputMode === 'webcam' ? 'Live Monitoring Feed' : 'Static File Analysis'}</span>
                      {driverState !== 'ACTIVE' && (isCameraActive || inputMode === 'upload') && (
                        <span className="ml-1 text-[10px] bg-red-500/20 text-red-300 border border-red-500/30 px-2 py-0.5 rounded-full font-bold uppercase alert-flash">
                          {driverState}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3">
                      {isCameraActive && (
                        <button onClick={stopCamera}
                          className="text-[10px] px-2.5 py-1 bg-red-500/10 hover:bg-red-500/20 text-red-400 border border-red-500/30 rounded-lg font-bold transition-all uppercase tracking-wide mr-2">
                          Stop Feed
                        </button>
                      )}
                      <div className="flex gap-1 bg-white/[0.03] border border-white/[0.06] p-0.5 rounded-xl">
                        {['webcam', 'upload'].map(m => (
                          <button key={m}
                            onClick={() => {
                              if (m === 'upload') {
                                setInputMode('upload');
                                stopCamera();
                              } else {
                                setInputMode('webcam');
                                setUploadedImageName(null);
                                setStaticImage(null);
                                setStaticLandmarks(null);
                                setDriverState('ACTIVE');
                                setLandmarkCount(0);
                                setPresence('👤 Alone');
                              }
                            }}
                            className={`text-[10px] px-2.5 py-1 rounded-lg font-bold transition-all capitalize ${
                              inputMode === m
                                ? 'bg-blue-600 text-white shadow-lg'
                                : 'text-textSecondary hover:text-white'
                            }`}
                          >
                            {m === 'webcam' ? '📷 Webcam' : '📁 Upload'}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Video canvas */}
                  <div className="flex-1 bg-black/50 relative flex items-center justify-center overflow-hidden">
                    <video ref={videoRef} autoPlay playsInline muted className="hidden" />
                    <canvas ref={canvasRef} className="max-w-full max-h-full object-contain" />

                    {/* Touch-to-Start Camera Overlay */}
                    {!isCameraActive && inputMode === 'webcam' && (
                      <div onClick={startCamera}
                        className="absolute inset-0 bg-darkBg/95 flex flex-col items-center justify-center cursor-pointer group hover:bg-darkBg/90 transition-all p-6 text-center z-10">
                        <div className="w-20 h-20 rounded-full bg-blue-500/10 border border-blue-500/25 flex items-center justify-center mb-4 transition-all duration-300 group-hover:scale-110 group-hover:bg-blue-500/20 group-hover:border-blue-500/50 shadow-lg glow-blue">
                          <Video className="w-10 h-10 text-blue-400 group-hover:text-blue-300" />
                        </div>
                        <h3 className="text-[16px] font-bold tracking-wide text-white group-hover:text-blue-300 transition-colors">Start Webcam Monitoring</h3>
                        <p className="text-[11px] text-textSecondary max-w-xs mt-1.5 leading-relaxed">
                          Click or tap to activate camera and start real-time multi-modal drowsiness tracking.
                        </p>
                      </div>
                    )}

                    {/* Upload zone */}
                    {inputMode === 'upload' && !uploadedImageName && (
                      <div onDragOver={handleDragOver} onDrop={handleDrop} onClick={() => fileInputRef.current.click()}
                        className="absolute inset-0 bg-darkBg/95 flex flex-col items-center justify-center cursor-pointer group hover:bg-darkBg/90 transition-all p-6 text-center z-10 border-2 border-dashed border-white/10 hover:border-blue-500/40">
                        <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/25 flex items-center justify-center mb-4 transition-all duration-300 group-hover:scale-110 group-hover:bg-blue-500/20 group-hover:border-blue-500/50 shadow-lg glow-blue">
                          <UploadCloud className="w-8 h-8 text-blue-400" />
                        </div>
                        <h3 className="text-[15px] font-bold tracking-wide text-white group-hover:text-blue-300 transition-colors">Upload Image or Video</h3>
                        <p className="text-[11px] text-textSecondary max-w-xs mt-1.5 leading-relaxed">
                          Drag and drop or click to upload a photo/video for offline drowsiness classification.
                        </p>
                        <p className="text-[9px] text-textSecondary/50 mt-1 uppercase tracking-wider">Supports JPG · PNG · MP4 · WEBM</p>
                        <input type="file" ref={fileInputRef} accept="image/*,video/*" className="hidden"
                          onChange={(e) => { if (e.target.files[0]) processUploadedFile(e.target.files[0]); }} />
                      </div>
                    )}

                    {/* Uploaded image status bar overlay */}
                    {inputMode === 'upload' && uploadedImageName && (
                      <div className="absolute bottom-4 left-4 right-4 bg-darkBg/90 backdrop-blur border border-white/[0.08] px-4 py-3 rounded-xl flex items-center justify-between z-25 shadow-xl">
                        <div className="flex items-center gap-2.5">
                          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center">
                            <CheckCircle className="w-4 h-4 text-emerald-400" />
                          </div>
                          <div className="text-left">
                            <p className="text-[11px] font-semibold text-white">Analysis Complete</p>
                            <p className="text-[10px] text-textSecondary truncate max-w-[150px] md:max-w-xs">{uploadedImageName}</p>
                          </div>
                        </div>
                        <button
                          onClick={() => {
                            setUploadedImageName(null);
                            setStaticImage(null);
                            setStaticLandmarks(null);
                            setDriverState('ACTIVE');
                            setLandmarkCount(0);
                            setPresence('👤 Alone');
                            const w = canvasRef.current.width  = 640;
                            const h = canvasRef.current.height = 480;
                            const ctx = canvasRef.current.getContext("2d");
                            ctx.clearRect(0, 0, w, h);
                          }}
                          className="px-3 py-1.5 bg-white/[0.05] hover:bg-white/[0.1] text-white text-[10px] font-semibold rounded-lg transition-all border border-white/[0.08]"
                        >
                          Clear &amp; Upload New
                        </button>
                      </div>
                    )}

                    {/* Alert overlay */}
                    {driverState !== 'ACTIVE' && (isCameraActive || (inputMode === 'upload' && uploadedImageName)) && (
                      <div className={`absolute inset-0 pointer-events-none z-10 flex flex-col items-center justify-center
                        ${driverState === 'DROWSY'
                          ? 'border-4 border-red-500 bg-red-950/30 alert-flash'
                          : 'border-4 border-orange-500 bg-orange-950/20 alert-flash'}`}>
                        <div className={`p-4 rounded-full mb-3 ${driverState === 'DROWSY' ? 'bg-red-500/20' : 'bg-orange-500/20'}`}>
                          <AlertOctagon className={`w-12 h-12 ${driverState === 'DROWSY' ? 'text-red-400' : 'text-orange-400'}`} />
                        </div>
                        <span className={`text-2xl font-black tracking-widest ${driverState === 'DROWSY' ? 'text-red-300' : 'text-orange-300'}`}>
                          ⚠ {driverState} ⚠
                        </span>
                        <span className="text-[12px] text-white/70 mt-1.5 font-medium">Stay alert — alarm triggered</span>
                      </div>
                    )}

                    {/* Camera error */}
                    {webcamError && !loading && (
                      <div className="absolute inset-0 bg-darkBg/95 flex flex-col items-center justify-center p-6 text-center">
                        <div className="w-14 h-14 rounded-2xl bg-orange-500/10 border border-orange-500/25 flex items-center justify-center mb-3">
                          <AlertOctagon className="w-7 h-7 text-orange-400" />
                        </div>
                        <p className="text-[13px] font-semibold text-orange-400 mb-1">Camera Unavailable</p>
                        <p className="text-[11px] text-textSecondary max-w-xs">{webcamError}</p>
                        <button onClick={startCamera}
                          className="mt-4 px-4 py-2 bg-blue-600 text-white text-[12px] font-semibold rounded-xl hover:bg-blue-500 transition-all">
                          Retry Camera
                        </button>
                      </div>
                    )}

                    {/* Loading */}
                    {loading && (
                      <div className="absolute inset-0 bg-darkBg flex flex-col items-center justify-center z-20">
                        <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/25 flex items-center justify-center mb-4 animate-pulse">
                          <RefreshCcw className="w-7 h-7 text-blue-400 animate-spin" />
                        </div>
                        <p className="text-[14px] font-semibold tracking-wide mb-1">Loading AI Models...</p>
                        <p className="text-[11px] text-textSecondary">MediaPipe FaceMesh · CNN · ST-GCN</p>
                        <div className="mt-4 w-48 h-1 rounded-full bg-white/5 overflow-hidden">
                          <div className="h-full rounded-full loading-shimmer w-full"></div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* RIGHT COLUMN: Details */}
              <div className="lg:col-span-5 xl:col-span-4 flex flex-col gap-4">

                {/* ── DRIVER STATE Banner ──────────────────────────────────────── */}
                <div className={`rounded-xl py-5 px-4 text-center text-white transition-all duration-300 ${
                  driverState === 'DROWSY'
                    ? 'bg-gradient-to-r from-red-500 to-rose-600 shadow-[0_4px_20px_rgba(239,68,68,0.25)]'
                    : driverState !== 'ACTIVE'
                    ? 'bg-gradient-to-r from-amber-500 to-orange-600 shadow-[0_4px_20px_rgba(245,158,11,0.25)]'
                    : 'bg-gradient-to-r from-blue-600 to-indigo-600 shadow-[0_4px_20px_rgba(37,99,235,0.25)]'
                }`}>
                  <p className="text-[11px] uppercase tracking-widest font-semibold opacity-75">Driver State</p>
                  <p className="text-[28px] font-black tracking-wide uppercase mt-1">
                    {driverState === 'ACTIVE' ? 'NORMAL' : driverState}
                  </p>
                </div>

                {/* ── Metrics Grid (6 Items) ───────────────────────────────────── */}
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { label: 'EAR', val: ear.toFixed(3) },
                    { label: 'Blink Rate', val: `${blinkRate.toFixed(1)}/min` },
                    { label: 'Blinks', val: blinks },
                    { label: 'Yawns', val: yawns },
                    { label: 'Closure Duration', val: `${closureDuration.toFixed(2)}s` },
                    { label: 'Head Direction', val: headDir },
                  ].map(({ label, val }) => (
                    <div key={label} className="bg-white/[0.03] border border-white/[0.05] rounded-xl p-4">
                      <p className="text-[10px] text-textSecondary uppercase font-bold tracking-wide">{label}</p>
                      <p className="text-[18px] font-bold text-white mt-1">{val}</p>
                    </div>
                  ))}
                </div>

                {/* Divider */}
                <div className="border-t border-white/[0.06] my-1"></div>

                {/* ── States Grid (Emotion & Presence) ─────────────────────────── */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl p-4">
                    <p className="text-[10px] text-textSecondary uppercase font-bold tracking-wide">Emotion</p>
                    <p className="text-[16px] font-bold text-white mt-1.5 flex items-center gap-1.5">
                      {emotion}
                    </p>
                  </div>
                  <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl p-4">
                    <p className="text-[10px] text-textSecondary uppercase font-bold tracking-wide">Presence</p>
                    <p className="text-[16px] font-bold text-white mt-1.5 flex items-center gap-1.5">
                      {presence}
                    </p>
                  </div>
                </div>

                {/* ── Fatigue Score Card ────────────────────────────────────────── */}
                <div className="bg-white/[0.03] border border-white/[0.05] rounded-xl p-4 flex flex-col justify-between">
                  <div>
                    <p className="text-[10px] text-textSecondary uppercase font-bold tracking-wide">Driver Fatigue Score</p>
                    <p className="text-[24px] font-black text-white mt-1">{fatigue.toFixed(1)}%</p>
                  </div>
                  <div className="mt-2.5 flex items-center gap-2">
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase ${
                      fatigue > 60 ? 'bg-red-500/20 text-red-300' :
                      fatigue > 30 ? 'bg-orange-500/20 text-orange-300' : 'bg-emerald-500/20 text-emerald-300'
                    }`}>
                      {fatigue > 60 ? 'Critical' : fatigue > 30 ? 'Moderate' : 'Safe'}
                    </span>
                    <span className="text-[10px] text-textSecondary font-semibold">
                      ↑ {fatigue.toFixed(1)}%
                    </span>
                  </div>
                </div>

              </div>

            </motion.div>
          )}

          {/* ════ HISTORY TAB ══════════════════════════════════════════════════ */}
          {activeTab === 'history' && (
            <motion.div key="history"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="p-6 max-w-5xl mx-auto w-full">
              <div className="glass-panel-bright p-6 rounded-2xl">
                <div className="flex justify-between items-center border-b border-white/[0.06] pb-4 mb-6">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-violet-500/15 border border-violet-500/25 flex items-center justify-center">
                      <Database className="w-4.5 h-4.5 text-violet-400" />
                    </div>
                    <div>
                      <h2 className="text-[16px] font-bold">Session Registry</h2>
                      <p className="text-[11px] text-textSecondary">{sessionHistory.length} sessions recorded</p>
                    </div>
                  </div>
                  <button onClick={loadHistory}
                    className="w-9 h-9 flex items-center justify-center border border-white/[0.08] hover:bg-white/5 rounded-xl transition-all">
                    <RefreshCcw className="w-4 h-4 text-textSecondary" />
                  </button>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-white/[0.05]">
                        {['Timestamp','Duration','Blinks','Yawns','Peak Fatigue','Status'].map(h => (
                          <th key={h} className="pb-3 text-[11px] text-textSecondary font-semibold uppercase tracking-wider">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/[0.03]">
                      {sessionHistory.length === 0 ? (
                        <tr><td colSpan="6" className="py-12 text-center text-textSecondary text-[12px]">
                          <div className="flex flex-col items-center gap-2 opacity-50">
                            <Database className="w-8 h-8" />
                            <span>No sessions yet. Click "Log Session" after a drive.</span>
                          </div>
                        </td></tr>
                      ) : [...sessionHistory].reverse().map((s, idx) => (
                        <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                          <td className="py-3.5 font-mono text-[11px] text-textSecondary">{s.timestamp}</td>
                          <td className="py-3.5 text-[12px]">
                            <span className="flex items-center gap-1 text-textSecondary">
                              <Clock className="w-3 h-3" />{Math.floor(s.duration_sec/60)}m {s.duration_sec%60}s
                            </span>
                          </td>
                          <td className="py-3.5 text-center font-bold text-[13px]">{s.blinks_count}</td>
                          <td className="py-3.5 text-center font-bold text-[13px]">{s.yawns_count}</td>
                          <td className={`py-3.5 text-center font-black text-[13px] ${s.max_fatigue > 60 ? 'text-red-400' : s.max_fatigue > 30 ? 'text-orange-400' : 'text-emerald-400'}`}>
                            {s.max_fatigue.toFixed(1)}%
                          </td>
                          <td className="py-3.5">
                            <span className={`text-[10px] px-2.5 py-1 rounded-full font-bold uppercase ${s.status === 'SAFE' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
                              {s.status === 'SAFE' ? '✓ Safe' : '⚠ Alert'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>
          )}

          {/* ════ SETTINGS TAB ════════════════════════════════════════════════ */}
          {activeTab === 'settings' && (
            <motion.div key="settings"
              initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
              className="p-6 max-w-xl mx-auto w-full">
              <div className="glass-panel-bright p-6 rounded-2xl space-y-6">
                <div className="flex items-center gap-3 border-b border-white/[0.06] pb-5">
                  <div className="w-9 h-9 rounded-xl bg-blue-500/15 border border-blue-500/25 flex items-center justify-center">
                    <Sliders className="w-4 h-4 text-blue-400" />
                  </div>
                  <div>
                    <h2 className="text-[16px] font-bold">Detection Calibration</h2>
                    <p className="text-[11px] text-textSecondary">Tune thresholds to your face geometry</p>
                  </div>
                </div>

                <div className="space-y-7">
                  {[
                    { label: 'Eye Aspect Ratio (EAR)', desc: 'Lower = more sensitive to eye closure', val: earThreshold, set: setEarThreshold, min: 0.10, max: 0.40, step: 0.01, accent: 'text-blue-400', bg: 'bg-blue-400/10 border-blue-400/20' },
                    { label: 'Mouth Aspect Ratio (MAR)', desc: 'Higher = less sensitive to yawning', val: marThreshold, set: setMarThreshold, min: 0.40, max: 0.80, step: 0.01, accent: 'text-orange-400', bg: 'bg-orange-400/10 border-orange-400/20' },
                    { label: 'Head Pose Tolerance (Yaw)', desc: 'Allowable gaze deviation from forward', val: yawTolerance, set: setYawTolerance, min: 0.05, max: 0.30, step: 0.01, accent: 'text-violet-400', bg: 'bg-violet-400/10 border-violet-400/20' },
                  ].map(({ label, desc, val, set, min, max, step, accent, bg }) => (
                    <div key={label} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <div>
                          <p className="text-[13px] font-semibold">{label}</p>
                          <p className="text-[10px] text-textSecondary mt-0.5">{desc}</p>
                        </div>
                        <span className={`px-2.5 py-1 ${bg} ${accent} rounded-lg font-bold text-[12px] border font-mono`}>{val.toFixed(2)}</span>
                      </div>
                      <input type="range" min={min} max={max} step={step} value={val}
                        onChange={(e) => set(parseFloat(e.target.value))} />
                    </div>
                  ))}

                  <div className="bg-white/[0.025] rounded-xl p-4 border border-white/[0.06] text-[11px] text-textSecondary space-y-1.5">
                    <p className="text-white font-semibold text-[12px] mb-2">Current Sensitivity</p>
                    <p>👁 Drowsy: eyes closed for <strong className="text-white">{CONSEC_FRAMES_EAR} frames</strong></p>
                    <p>🥱 Yawn: mouth open for <strong className="text-white">{CONSEC_FRAMES_MAR} frames</strong></p>
                    <p>👀 Distraction: head turned for <strong className="text-white">{CONSEC_FRAMES_YAW} frames</strong></p>
                  </div>

                  <div className="bg-white/[0.025] rounded-xl p-4 border border-white/[0.06] space-y-3">
                    <p className="text-white font-semibold text-[12px]">API Settings</p>
                    <p className="text-[10px] text-textSecondary">Set your backend API URL if hosted separately (e.g. Vercel frontend, local backend):</p>
                    <div className="flex gap-2">
                      <input 
                        type="text" 
                        value={backendUrl} 
                        onChange={(e) => {
                          const val = e.target.value;
                          setBackendUrl(val);
                          localStorage.setItem('drowsishield_backend_url', val);
                        }}
                        className="flex-1 bg-black/40 border border-white/10 rounded-xl px-3 py-2 text-[12px] text-white focus:outline-none focus:border-blue-500/50 font-mono"
                        placeholder="http://localhost:5000"
                      />
                      <button 
                        onClick={() => {
                          const val = window.location.origin.replace(/\/$/, '');
                          setBackendUrl(val);
                          localStorage.removeItem('drowsishield_backend_url');
                          addLog("Backend API URL reset to default origin.", "info");
                        }}
                        className="px-3 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-[11px] font-semibold transition-all active:scale-95"
                      >
                        Reset
                      </button>
                    </div>
                  </div>

                  <button onClick={() => { setEarThreshold(0.25); setMarThreshold(0.60); setYawTolerance(0.15); addLog("Settings restored to factory defaults.", "info"); }}
                    className="w-full py-2.5 border border-white/[0.08] hover:bg-white/[0.04] rounded-xl font-semibold text-[12px] text-textSecondary hover:text-white transition-all">
                    ↺ Restore Factory Defaults
                  </button>
                </div>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>
    </div>
  );
}

export default Dashboard;
