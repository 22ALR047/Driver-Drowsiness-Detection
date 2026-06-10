/**
 * AudioAlarm.js — DrowsiShield Elegant Alert System
 * Clean, pleasant chime-based alerts using sine waves with smooth envelopes.
 */

// ── WAV builder ───────────────────────────────────────────────────────────────
function _makeWAV(getSample, durationMs, sampleRate = 44100) {
  const numSamples = Math.floor(sampleRate * durationMs / 1000);
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view   = new DataView(buffer);

  const ws = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
  ws(0, 'RIFF'); view.setUint32(4, 36 + numSamples * 2, true);
  ws(8, 'WAVE'); ws(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  ws(36, 'data'); view.setUint32(40, numSamples * 2, true);

  for (let i = 0; i < numSamples; i++) {
    const t   = i / sampleRate;
    const dur = durationMs / 1000;
    const s   = Math.max(-32768, Math.min(32767, Math.round(getSample(t, dur) * 32767)));
    view.setInt16(44 + i * 2, s, true);
  }

  const bytes = new Uint8Array(buffer);
  let bin = '';
  for (let i = 0; i < bytes.length; i += 8192)
    bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return 'data:audio/wav;base64,' + btoa(bin);
}

// ── Bell-like chime using sine + exponential decay ────────────────────────────
function _chime(freq, durationMs, vol = 0.7) {
  return _makeWAV((t, dur) => {
    const attack  = Math.min(1, t / 0.01);                  // 10ms attack
    const decay   = Math.exp(-t * 5);                        // natural bell decay
    const env     = attack * decay * vol;
    return Math.sin(2 * Math.PI * freq * t) * env
         + Math.sin(2 * Math.PI * freq * 2 * t) * env * 0.15; // soft 2nd harmonic
  }, durationMs);
}

// ── Pre-generate sounds ───────────────────────────────────────────────────────
// Drowsy : three rising chimes  (C5 → E5 → G5)
// Yawning: two gentle tones     (A4 → E4)
// Attention: single soft ping   (B4)
// Quick  : tiny confirmation    (G4)
const SOUNDS = {
  drowsy_1:   _chime(523, 600, 0.75),  // C5
  drowsy_2:   _chime(659, 600, 0.80),  // E5
  drowsy_3:   _chime(784, 700, 0.85),  // G5
  yawn_1:     _chime(440, 600, 0.70),  // A4
  yawn_2:     _chime(330, 700, 0.65),  // E4
  attention:  _chime(494, 600, 0.68),  // B4
  quick:      _chime(392, 300, 0.50),  // G4
};

// ── Playback ──────────────────────────────────────────────────────────────────
function _play(dataURL) {
  try {
    const a = new Audio(dataURL);
    a.volume = 1.0;
    a.play().catch(() => {});
  } catch (e) {}
}

// ── State ─────────────────────────────────────────────────────────────────────
let alarmInterval = null;
let isMuted       = false;

// ── Speech ────────────────────────────────────────────────────────────────────
function _speak(text) {
  if (!('speechSynthesis' in window) || isMuted) return;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95; u.pitch = 1.0; u.volume = 1.0;
    window.speechSynthesis.speak(u);
  } catch (e) {}
}

// ── Patterns ──────────────────────────────────────────────────────────────────
const _patterns = {
  // Three rising chimes — clear but not harsh
  drowsy: () => {
    _play(SOUNDS.drowsy_1);
    setTimeout(() => _play(SOUNDS.drowsy_2), 350);
    setTimeout(() => _play(SOUNDS.drowsy_3), 700);
  },
  // Two descending tones — gentle caution
  yawning: () => {
    _play(SOUNDS.yawn_1);
    setTimeout(() => _play(SOUNDS.yawn_2), 400);
  },
  // Single soft ping
  attention: () => {
    _play(SOUNDS.attention);
  },
};

// ── Public API ────────────────────────────────────────────────────────────────
export const startAlarm = (type = 'drowsy') => {
  if (isMuted || alarmInterval) return;

  const pattern = _patterns[type] || _patterns.drowsy;
  const rate    = type === 'drowsy' ? 1400 : 2000; // repeat interval (ms)

  pattern(); // play immediately

  if (type === 'drowsy') {
    setTimeout(() => _speak('Please stay alert. Drowsiness detected.'), 1200);
  } else if (type === 'yawning') {
    setTimeout(() => _speak('You are yawning. Consider taking a short break.'), 900);
  } else if (type === 'attention') {
    setTimeout(() => _speak('Please look forward. Keep your eyes on the road.'), 700);
  }

  alarmInterval = setInterval(() => {
    if (isMuted) { stopAlarm(); return; }
    pattern();
  }, rate);
};

export const stopAlarm = () => {
  if (alarmInterval) { clearInterval(alarmInterval); alarmInterval = null; }
  try { if ('speechSynthesis' in window) window.speechSynthesis.cancel(); } catch (e) {}
};

export const triggerQuickBeep = () => {
  if (isMuted) return;
  _play(SOUNDS.quick);
};

export const initAudio = () => true;

export const setMuted = (muted) => {
  isMuted = muted;
  if (muted) stopAlarm();
};

export const getMuted      = () => isMuted;
export const isAlarmActive = () => alarmInterval !== null;
