const {
  LoudnessMeterEngine,
  STREAMING_PRESETS,
  analyzeAudioBufferOffline,
  computeKWeightingCoeffs,
  formatLUFS,
  formatDB,
} = window.ULM_DSP;

// ================= STATE =================
let audioCtx = null;
let engine = new LoudnessMeterEngine(48000, 2);

let currentMode = 'standby'; // 'standby' | 'live' | 'system' | 'file' | 'tone'
let currentPresetKey = 'spotify';
let timeWindowSec = 30;
let graphPaused = false;
let barMeterMode = 'M'; // 'M' (Momentary bar) | 'S' (Short-Term bar)
let ballisticsSpeed = 'standard'; // 'fast' | 'standard' | 'smooth'
let colorTheme = 'classic'; // 'classic' | 'cyan' | 'emerald'

const THEME_PALETTES = {
  classic: {
    barMain: '#5887da',
    barOver: '#d95852',
    curveFill: 'rgba(88, 135, 218, 0.45)',
    curveStroke: '#6b9be8',
    overFill: 'rgba(217, 88, 82, 0.55)',
    overStroke: '#ef6b65',
  },
  cyan: {
    barMain: '#06b6d4',
    barOver: '#f59e0b',
    curveFill: 'rgba(6, 182, 212, 0.42)',
    curveStroke: '#22d3ee',
    overFill: 'rgba(245, 158, 11, 0.52)',
    overStroke: '#fbbf24',
  },
  emerald: {
    barMain: '#10b981',
    barOver: '#ef4444',
    curveFill: 'rgba(16, 185, 129, 0.42)',
    curveStroke: '#34d399',
    overFill: 'rgba(239, 68, 68, 0.55)',
    overStroke: '#f87171',
  },
};

// Pro Audition States
let normPreviewEnabled = false;
let monoCheckEnabled = false;

// Graph layer toggles (S, M, T, D, G)
const layers = {
  S: true,  // Short-Term curve
  M: false, // Momentary curve
  T: true,  // True Peak markers
  D: false, // Dynamics / A-B Ghost Comparison curve
  G: true,  // Grid & Target line
};

// Live history buffer & Histogram bins (-60 LUFS to 0 LUFS, 120 bins of 0.5 LU)
const HIST_BINS = 120;
let liveHistory = [];
let liveHistogram = new Float64Array(HIST_BINS);

// A / B Comparison Slots
let activeSlotKey = 'A';
const slots = {
  A: {
    audioBuffer: null,
    fileName: '',
    analysis: null,
    histogram: new Float64Array(HIST_BINS),
    pauseOffsetSec: 0,
  },
  B: {
    audioBuffer: null,
    fileName: '',
    analysis: null,
    histogram: new Float64Array(HIST_BINS),
    pauseOffsetSec: 0,
  },
};

// Active WebAudio nodes
let activeStream = null;
let sourceNode = null;
let processorNode = null;
let zeroGainNode = null;
let toneOscNode = null;
let toneGainNode = null;

// File playback nodes
let fileSourceNode = null;
let playbackGainNode = null;
let monoSumNode = null;
let fileIsPlaying = false;
let filePlayStartCtxTime = 0;

// Graph hover & drag-scrub state
let hoverPoint = null;
let isDraggingSeek = false;
let dragSeekZone = null; // 'main' | 'minimap'
let lastGraphView = { startTime: 0, endTime: 30, totalDuration: 30, plotLeft: 152, plotRight: 800, miniLeft: 116, miniW: 684 };

// ================= 60FPS STUDIO METER BALLISTICS & DISPLAY SMOOTHING =================
let lastTickTime = performance.now();
let lastReadoutUpdateTime = 0;

const ballistics = {
  momentary: -58,
  shortTerm: -58,
  integrated: -Infinity,
  lra: 0,
  psr: -Infinity,
  plr: -Infinity,
  correlation: 1.0,
  reset() {
    this.momentary = -58;
    this.shortTerm = -58;
    this.integrated = -Infinity;
    this.lra = 0;
    this.psr = -Infinity;
    this.plr = -Infinity;
    this.correlation = 1.0;
  },
};

function updateBallistics(snap, dt) {
  const clampedDt = Math.min(0.1, Math.max(0.001, dt));
  const mult = ballisticsSpeed === 'fast' ? 0.5 : ballisticsSpeed === 'smooth' ? 1.65 : 1.0;

  const smoothValue = (current, target, tauUp, tauDown, maxRatePerSec) => {
    if (!isFinite(target)) return current;
    if (!isFinite(current)) return target;
    const tau = (target > current ? tauUp : tauDown) * mult;
    const alpha = 1 - Math.exp(-clampedDt / tau);
    let delta = (target - current) * alpha;
    const maxDelta = (maxRatePerSec / mult) * clampedDt;
    if (Math.abs(delta) > maxDelta) {
      delta = Math.sign(delta) * maxDelta;
    }
    return current + delta;
  };

  // 1. Momentary Bar Gauge: damped attack (200ms), smooth studio release (480ms), max 15 LU/sec slew
  const targetMom = isFinite(snap.momentary) && snap.momentary > -58 ? snap.momentary : -58;
  ballistics.momentary = smoothValue(ballistics.momentary, targetMom, 0.20, 0.48, 15.0);

  // 2. Short-Term: calm 380ms time constant, max 8 LU/sec slew
  if (isFinite(snap.shortTerm) && snap.shortTerm > -80) {
    if (ballistics.shortTerm <= -57.5) {
      ballistics.shortTerm = snap.shortTerm;
    } else {
      ballistics.shortTerm = smoothValue(ballistics.shortTerm, snap.shortTerm, 0.36, 0.45, 8.0);
    }
  } else {
    ballistics.shortTerm = smoothValue(ballistics.shortTerm, -58, 0.45, 0.45, 10.0);
  }

  // 3. Integrated LUFS: calm 450ms time constant
  if (isFinite(snap.integrated) && snap.integrated > -80) {
    ballistics.integrated = smoothValue(ballistics.integrated, snap.integrated, 0.45, 0.45, 6.0);
  } else {
    ballistics.integrated = -Infinity;
  }

  // 4. LRA: calm 550ms time constant
  ballistics.lra = smoothValue(ballistics.lra, snap.lra || 0, 0.55, 0.55, 4.0);

  // 5. PSR & PLR: calm 500ms time constant so LU numbers never jump
  if (isFinite(snap.psr) && snap.psr > 0) {
    ballistics.psr = smoothValue(ballistics.psr, snap.psr, 0.50, 0.60, 5.0);
  } else {
    ballistics.psr = -Infinity;
  }

  if (isFinite(snap.plr) && snap.plr > 0) {
    ballistics.plr = smoothValue(ballistics.plr, snap.plr, 0.50, 0.60, 5.0);
  } else {
    ballistics.plr = -Infinity;
  }

  // 6. Phase Correlation: responsive 140ms time constant so phase graph moves livelily
  const rawCorr = monoCheckEnabled ? 1.0 : isFinite(snap.correlation) ? snap.correlation : 1.0;
  ballistics.correlation = smoothValue(ballistics.correlation, rawCorr, 0.14, 0.16, 4.5);
}

// ================= DOM ELEMENTS =================
const elStatusText = document.getElementById('statusText');
const btnLiveInput = document.getElementById('btnLiveInput');
const deviceSelect = document.getElementById('deviceSelect');
const btnSystemAudio = document.getElementById('btnSystemAudio');
const fileInput = document.getElementById('fileInput');
const lblFilesBtn = document.getElementById('lblFilesBtn');
const btnTestTone = document.getElementById('btnTestTone');

const btnNormPreview = document.getElementById('btnNormPreview');
const btnMonoCheck = document.getElementById('btnMonoCheck');

const presetSelect = document.getElementById('presetSelect');
const btnReset = document.getElementById('btnReset');
const btnMiniMode = document.getElementById('btnMiniMode');
const btnPiP = document.getElementById('btnPiP');
const btnAutoScale = document.getElementById('btnAutoScale');
const btnIntLra = document.getElementById('btnIntLra');

// Export Menu DOM
const btnExport = document.getElementById('btnExport');
const exportMenu = document.getElementById('exportMenu');
const btnExportPNG = document.getElementById('btnExportPNG');
const btnExportCSV = document.getElementById('btnExportCSV');
const btnExportJSON = document.getElementById('btnExportJSON');

// Settings Menu DOM
const btnSettings = document.getElementById('btnSettings');
const settingsMenu = document.getElementById('settingsMenu');
const btnCloseSettings = document.getElementById('btnCloseSettings');
const inpCustomTarget = document.getElementById('inpCustomTarget');
const inpCustomCeiling = document.getElementById('inpCustomCeiling');
const selBallistics = document.getElementById('selBallistics');
const selTheme = document.getElementById('selTheme');
const btnClearAllSlots = document.getElementById('btnClearAllSlots');

// Readout DOM (5-row stack + 3 pills + Phase Correlation)
const valShortTerm = document.getElementById('valShortTerm');
const valIntegrated = document.getElementById('valIntegrated');
const valLRA = document.getElementById('valLRA');
const valPSR = document.getElementById('valPSR');
const valPLR = document.getElementById('valPLR');

const valMomentaryMax = document.getElementById('valMomentaryMax');
const valShortTermMax = document.getElementById('valShortTermMax');
const valTruePeakMax = document.getElementById('valTruePeakMax');
const verdictText = document.getElementById('verdictText');

const valPhaseCorr = document.getElementById('valPhaseCorr');
const phaseFill = document.getElementById('phaseFill');
const phaseMinTick = document.getElementById('phaseMinTick');
const phaseThumb = document.getElementById('phaseThumb');

// A/B & Graph DOM
const btnSlotA = document.getElementById('btnSlotA');
const btnSlotB = document.getElementById('btnSlotB');
const dropTitle = document.getElementById('dropTitle');

const windowSelector = document.getElementById('windowSelector');
const btnPauseGraph = document.getElementById('btnPauseGraph');
const lblGraphMode = document.getElementById('lblGraphMode');
const historyCanvas = document.getElementById('historyCanvas');
const barMeterCanvas = document.getElementById('barMeterCanvas');
const dropOverlay = document.getElementById('dropOverlay');

const btnPlayPause = document.getElementById('btnPlayPause');
const iconPlay = document.getElementById('iconPlay');
const btnStopFile = document.getElementById('btnStopFile');
const fileMetaInfo = document.getElementById('fileMetaInfo');

const btnFloatAudio = document.getElementById('btnFloatAudio');
const btnFloatPeaks = document.getElementById('btnFloatPeaks');
const btnFloatTarget = document.getElementById('btnFloatTarget');

// ================= SLOT & HISTOGRAM HELPERS =================
function getSlot(key = activeSlotKey) {
  return slots[key];
}

function getOtherSlot() {
  return slots[activeSlotKey === 'A' ? 'B' : 'A'];
}

function addHistogramSample(histArray, lufsValue) {
  if (!isFinite(lufsValue) || lufsValue < -60 || lufsValue > 0) return;
  const binIdx = Math.min(HIST_BINS - 1, Math.max(0, Math.floor(((lufsValue + 60) / 60) * HIST_BINS)));
  histArray[binIdx] += 1;
}

// ================= STREAMING NORMALIZATION GAIN CALCULATOR =================
function computeNormalizationGainDB(snap) {
  const preset = STREAMING_PRESETS[currentPresetKey];
  if (!snap || !isFinite(snap.integrated) || snap.integrated < -70) return 0;
  let gainDB = preset.targetLUFS - snap.integrated;
  if (gainDB > 0 && isFinite(snap.maxTruePeak)) {
    const headroomDB = preset.maxTruePeak - snap.maxTruePeak;
    gainDB = Math.min(gainDB, Math.max(0, headroomDB));
  }
  return gainDB;
}

function applyRealtimePlaybackRouting() {
  if (!audioCtx || !playbackGainNode) return;
  const snap = getActiveSnapshot();
  const gainDB = normPreviewEnabled ? computeNormalizationGainDB(snap) : 0;
  const linearGain = Math.pow(10, gainDB / 20);
  playbackGainNode.gain.setTargetAtTime(linearGain, audioCtx.currentTime, 0.03);
}

// ================= AUDIO CONTEXT & GRAPH SETUP =================
function ensureAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    audioCtx = new AudioContextClass({ latencyHint: 'interactive' });
    engine.setConfig(audioCtx.sampleRate, 2);
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

function stopAllActiveAudio(keepFileLoaded = false) {
  if (fileSourceNode) {
    try {
      fileSourceNode.onended = null;
      fileSourceNode.stop();
      fileSourceNode.disconnect();
    } catch (_) {}
    fileSourceNode = null;
  }
  if (playbackGainNode) {
    try {
      playbackGainNode.disconnect();
    } catch (_) {}
    playbackGainNode = null;
  }
  if (monoSumNode) {
    try {
      monoSumNode.disconnect();
    } catch (_) {}
    monoSumNode = null;
  }
  fileIsPlaying = false;
  updatePlayPauseIcon();

  if (toneOscNode) {
    try {
      toneOscNode.stop();
      toneOscNode.disconnect();
    } catch (_) {}
    toneOscNode = null;
  }
  if (toneGainNode) {
    try {
      toneGainNode.disconnect();
    } catch (_) {}
    toneGainNode = null;
  }

  if (sourceNode) {
    try {
      sourceNode.disconnect();
    } catch (_) {}
    sourceNode = null;
  }
  if (processorNode) {
    try {
      processorNode.onaudioprocess = null;
      processorNode.disconnect();
    } catch (_) {}
    processorNode = null;
  }
  if (zeroGainNode) {
    try {
      zeroGainNode.disconnect();
    } catch (_) {}
    zeroGainNode = null;
  }

  if (activeStream) {
    activeStream.getTracks().forEach((t) => t.stop());
    activeStream = null;
  }

  if (!keepFileLoaded) {
    btnLiveInput.classList.remove('active');
    btnSystemAudio.classList.remove('active');
    btnTestTone.classList.remove('active');
  }
}

function attachRealtimeProcessor(inputNode, passthroughToSpeakers = false) {
  const ctx = ensureAudioContext();
  if (processorNode) {
    try {
      processorNode.onaudioprocess = null;
      processorNode.disconnect();
    } catch (_) {}
    processorNode = null;
  }
  if (zeroGainNode) {
    try {
      zeroGainNode.disconnect();
    } catch (_) {}
    zeroGainNode = null;
  }
  processorNode = ctx.createScriptProcessor(2048, 2, 2);
  zeroGainNode = ctx.createGain();
  zeroGainNode.gain.value = 0;

  let lastHistoryPushSec = -1;

  processorNode.onaudioprocess = (event) => {
    const inBuf = event.inputBuffer;
    const ch0 = inBuf.getChannelData(0);
    const ch1 = inBuf.numberOfChannels > 1 ? inBuf.getChannelData(1) : ch0;

    engine.process([ch0, ch1]);

    const snap = engine.getSnapshot();
    if (snap.durationSec - lastHistoryPushSec >= 0.095) {
      lastHistoryPushSec = snap.durationSec;
      addHistogramSample(liveHistogram, isFinite(snap.shortTerm) ? snap.shortTerm : snap.momentary);
      if (!graphPaused) {
        liveHistory.push({
          timeSec: snap.durationSec,
          momentary: snap.momentary,
          shortTerm: snap.shortTerm,
          integrated: snap.integrated,
          truePeakL: snap.truePeakL,
          truePeakR: snap.truePeakR,
          psr: snap.psr,
          correlation: snap.correlation,
        });
        if (liveHistory.length > 9000) {
          liveHistory.shift();
        }
      }
    }
  };

  inputNode.connect(processorNode);
  processorNode.connect(zeroGainNode);
  zeroGainNode.connect(ctx.destination);

  if (passthroughToSpeakers) {
    inputNode.connect(ctx.destination);
  }
}

// ================= 1. LIVE AUDIO DEVICE INPUT =================
async function populateAudioDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter((d) => d.kind === 'audioinput');
    const currentVal = deviceSelect.value;
    deviceSelect.innerHTML = '';

    const defaultOpt = document.createElement('option');
    defaultOpt.value = '';
    defaultOpt.textContent = '(A) STEREO ITU';
    deviceSelect.appendChild(defaultOpt);

    audioInputs.forEach((dev, idx) => {
      if (!dev.deviceId) return;
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `Audio Input #${idx + 1}`;
      deviceSelect.appendChild(opt);
    });

    if (currentVal && [...deviceSelect.options].some((o) => o.value === currentVal)) {
      deviceSelect.value = currentVal;
    }
  } catch (err) {
    console.warn('Could not enumerate audio devices:', err);
  }
}

async function startLiveDeviceInput() {
  if (currentMode === 'live') {
    stopAllActiveAudio();
    setStatus('standby', '');
    return;
  }

  try {
    stopAllActiveAudio();
    const ctx = ensureAudioContext();
    const selectedDeviceId = deviceSelect.value;

    const constraints = {
      audio: {
        deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: { ideal: 2 },
      },
      video: false,
    };

    activeStream = await navigator.mediaDevices.getUserMedia(constraints);
    await populateAudioDevices();

    engine.setConfig(ctx.sampleRate, 2);
    ballistics.reset();
    liveHistory = [];
    liveHistogram.fill(0);

    sourceNode = ctx.createMediaStreamSource(activeStream);
    attachRealtimeProcessor(sourceNode, false);

    btnLiveInput.classList.add('active');
    const trackLabel = activeStream.getAudioTracks()[0]?.label || 'Live Input';
    fileMetaInfo.textContent = trackLabel;
    setStatus('live', 'LIVE');
  } catch (err) {
    alert(`오디오 입력 장치에 접근할 수 없습니다: ${err.message}`);
  }
}

// ================= 2. SYSTEM / TAB AUDIO CAPTURE =================
async function startSystemAudioCapture() {
  if (currentMode === 'system') {
    stopAllActiveAudio();
    setStatus('standby', '');
    return;
  }

  if (!navigator.mediaDevices?.getDisplayMedia) {
    alert('현재 브라우저는 시스템/화면 오디오 캡처를 지원하지 않습니다. Chrome 또는 Edge를 사용해 주세요.');
    return;
  }

  try {
    stopAllActiveAudio();
    const ctx = ensureAudioContext();

    activeStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        suppressLocalAudioPlayback: false,
      },
    });

    const audioTracks = activeStream.getAudioTracks();
    if (audioTracks.length === 0) {
      activeStream.getTracks().forEach((t) => t.stop());
      activeStream = null;
      alert('화면/탭 공유 창 하단의 "오디오 공유(Share audio)" 체크박스를 켜주세요.');
      return;
    }

    audioTracks[0].onended = () => {
      stopAllActiveAudio();
      setStatus('standby', '');
    };

    engine.setConfig(ctx.sampleRate, 2);
    ballistics.reset();
    liveHistory = [];
    liveHistogram.fill(0);

    sourceNode = ctx.createMediaStreamSource(activeStream);
    attachRealtimeProcessor(sourceNode, false);

    btnSystemAudio.classList.add('active');
    fileMetaInfo.textContent = 'SYSTEM AUDIO';
    setStatus('system', 'SYS');
  } catch (err) {
    if (err.name !== 'NotAllowedError') {
      alert(`시스템 오디오 캡처 실패: ${err.message}`);
    }
  }
}

// ================= 3. REFERENCE -14.0 LUFS CALIBRATION TONE =================
function toggleReferenceTone() {
  if (currentMode === 'tone') {
    stopAllActiveAudio();
    setStatus('standby', '');
    return;
  }

  stopAllActiveAudio();
  const ctx = ensureAudioContext();
  engine.setConfig(ctx.sampleRate, 2);
  ballistics.reset();
  liveHistory = [];
  liveHistogram.fill(0);

  const freq = 1000;
  const w = (2 * Math.PI * freq) / ctx.sampleRate;
  const { stage1, stage2 } = computeKWeightingCoeffs(ctx.sampleRate);

  function biquadMagSq(b, omega) {
    const cos1 = Math.cos(omega);
    const cos2 = Math.cos(2 * omega);
    const sin1 = Math.sin(omega);
    const sin2 = Math.sin(2 * omega);
    const numRe = b.b0 + b.b1 * cos1 + b.b2 * cos2;
    const numIm = -(b.b1 * sin1 + b.b2 * sin2);
    const denRe = 1 + b.a1 * cos1 + b.a2 * cos2;
    const denIm = -(b.a1 * sin1 + b.a2 * sin2);
    return (numRe * numRe + numIm * numIm) / (denRe * denRe + denIm * denIm);
  }

  const H2 = biquadMagSq(stage1, w) * biquadMagSq(stage2, w);
  const targetEnergy = Math.pow(10, (-14.0 + 0.691) / 10);
  const amplitude = Math.sqrt(targetEnergy / H2);

  toneOscNode = ctx.createOscillator();
  toneOscNode.type = 'sine';
  toneOscNode.frequency.value = freq;

  toneGainNode = ctx.createGain();
  toneGainNode.gain.value = amplitude;

  toneOscNode.connect(toneGainNode);
  attachRealtimeProcessor(toneGainNode, true);
  toneOscNode.start();

  btnTestTone.classList.add('active');
  fileMetaInfo.textContent = '1 kHz (-14.0 LUFS)';
  setStatus('tone', 'TONE');
}

// ================= 4. FILE DRAG & DROP + A/B SLOT STORAGE & PLAYBACK =================
async function handleFileLoad(file) {
  if (!file) return;
  stopAllActiveAudio();

  try {
    const slot = getSlot();
    setStatus('file', `SLOT ${activeSlotKey}`);
    fileMetaInfo.textContent = `Loading ${file.name}...`;

    const ctx = ensureAudioContext();
    const arrayBuf = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuf);

    slot.audioBuffer = audioBuffer;
    slot.fileName = file.name;
    slot.pauseOffsetSec = 0;

    slot.analysis = analyzeAudioBufferOffline(audioBuffer);
    slot.histogram.fill(0);
    for (const pt of slot.analysis.timeline) {
      addHistogramSample(slot.histogram, isFinite(pt.shortTerm) ? pt.shortTerm : pt.momentary);
    }

    timeWindowSec = 0;
    windowSelector.value = '0';
    lastReadoutUpdateTime = 0;

    updateABButtonsUI();
    fileMetaInfo.textContent = `[${activeSlotKey}] ${file.name}`;
    setStatus('file', `SLOT ${activeSlotKey}`);
  } catch (err) {
    alert(`오디오 파일을 디코딩할 수 없습니다: ${err.message}`);
    setStatus('standby', '');
  }
}

function switchABSlot(targetKey) {
  if (targetKey !== 'A' && targetKey !== 'B') return;
  if (fileIsPlaying) {
    pauseFilePlayback();
  }
  activeSlotKey = targetKey;
  lastReadoutUpdateTime = 0;
  const slot = getSlot();
  if (slot.analysis) {
    currentMode = 'file';
    setStatus('file', `SLOT ${activeSlotKey}`);
    fileMetaInfo.textContent = `[${activeSlotKey}] ${slot.fileName}`;
  } else {
    fileMetaInfo.textContent = `SLOT ${activeSlotKey}: EMPTY (파일을 드롭하세요)`;
  }
  if (dropTitle) {
    dropTitle.textContent = `DROP AUDIO FILE INTO SLOT ${activeSlotKey}`;
  }
  if (lblFilesBtn) {
    lblFilesBtn.textContent = `FILES (${activeSlotKey})`;
  }
  updateABButtonsUI();
}

function updateABButtonsUI() {
  btnSlotA.classList.toggle('ab-active', activeSlotKey === 'A');
  btnSlotB.classList.toggle('ab-active', activeSlotKey === 'B');
  btnSlotA.classList.toggle('has-data', Boolean(slots.A.analysis));
  btnSlotB.classList.toggle('has-data', Boolean(slots.B.analysis));
}

function startFilePlayback(offsetSec) {
  const slot = getSlot();
  if (!slot.audioBuffer) return;
  const ctx = ensureAudioContext();

  if (fileSourceNode) {
    try {
      fileSourceNode.onended = null;
      fileSourceNode.stop();
      fileSourceNode.disconnect();
    } catch (_) {}
    fileSourceNode = null;
  }
  if (playbackGainNode) {
    try {
      playbackGainNode.disconnect();
    } catch (_) {}
    playbackGainNode = null;
  }
  if (monoSumNode) {
    try {
      monoSumNode.disconnect();
    } catch (_) {}
    monoSumNode = null;
  }
  if (processorNode) {
    try {
      processorNode.onaudioprocess = null;
      processorNode.disconnect();
    } catch (_) {}
    processorNode = null;
  }
  if (zeroGainNode) {
    try {
      zeroGainNode.disconnect();
    } catch (_) {}
    zeroGainNode = null;
  }

  const startOffset = offsetSec !== undefined ? offsetSec : slot.pauseOffsetSec;
  const clampedOffset = Math.max(0, Math.min(startOffset, slot.audioBuffer.duration - 0.05));

  fileSourceNode = ctx.createBufferSource();
  fileSourceNode.buffer = slot.audioBuffer;

  playbackGainNode = ctx.createGain();
  const gainDB = normPreviewEnabled ? computeNormalizationGainDB(slot.analysis?.summary) : 0;
  playbackGainNode.gain.value = Math.pow(10, gainDB / 20);

  fileSourceNode.connect(playbackGainNode);

  if (monoCheckEnabled && slot.audioBuffer.numberOfChannels >= 2) {
    const splitter = ctx.createChannelSplitter(2);
    const merger = ctx.createChannelMerger(2);
    const gainL = ctx.createGain();
    const gainR = ctx.createGain();
    gainL.gain.value = 0.5;
    gainR.gain.value = 0.5;

    playbackGainNode.connect(splitter);
    splitter.connect(gainL, 0);
    splitter.connect(gainR, 1);
    gainL.connect(merger, 0, 0);
    gainR.connect(merger, 0, 0);
    gainL.connect(merger, 0, 1);
    gainR.connect(merger, 0, 1);
    merger.connect(ctx.destination);
    monoSumNode = merger;
  } else {
    playbackGainNode.connect(ctx.destination);
  }

  slot.pauseOffsetSec = clampedOffset;
  filePlayStartCtxTime = ctx.currentTime - clampedOffset;

  fileSourceNode.onended = () => {
    if (fileIsPlaying) {
      fileIsPlaying = false;
      slot.pauseOffsetSec = 0;
      updatePlayPauseIcon();
    }
  };

  fileSourceNode.start(0, clampedOffset);
  fileIsPlaying = true;
  currentMode = 'file';
  updatePlayPauseIcon();
}

function pauseFilePlayback() {
  const slot = getSlot();
  if (!fileIsPlaying || !audioCtx || !slot.audioBuffer) return;
  slot.pauseOffsetSec = Math.min(
    slot.audioBuffer.duration,
    audioCtx.currentTime - filePlayStartCtxTime
  );
  if (fileSourceNode) {
    try {
      fileSourceNode.onended = null;
      fileSourceNode.stop();
      fileSourceNode.disconnect();
    } catch (_) {}
    fileSourceNode = null;
  }
  fileIsPlaying = false;
  updatePlayPauseIcon();
}

function stopFilePlayback() {
  pauseFilePlayback();
  getSlot().pauseOffsetSec = 0;
  lastReadoutUpdateTime = 0;
}

function getFilePlayheadSec() {
  const slot = getSlot();
  if (!slot.audioBuffer) return 0;
  if (fileIsPlaying && audioCtx) {
    return Math.min(slot.audioBuffer.duration, audioCtx.currentTime - filePlayStartCtxTime);
  }
  return slot.pauseOffsetSec;
}

function updatePlayPauseIcon() {
  if (fileIsPlaying) {
    iconPlay.innerHTML = '<rect x="5" y="4" width="5" height="16"/><rect x="14" y="4" width="5" height="16"/>';
  } else {
    iconPlay.innerHTML = '<polygon points="3 5 13 12 3 19 3 5"/><polygon points="13 5 23 12 13 19 13 5"/>';
  }
}

// ================= STATUS & READOUT FORMATTING =================
function setStatus(mode, label) {
  currentMode = mode;
  if (elStatusText) elStatusText.textContent = label;
}

function getActiveSnapshot() {
  const slot = getSlot();
  if (currentMode === 'file' && slot.analysis) {
    const summary = slot.analysis.summary;
    const tl = slot.analysis.timeline;
    const playSec = getFilePlayheadSec();

    // Interpolate from the pre-analyzed timeline whenever playing OR seeked to a position
    if ((fileIsPlaying || playSec > 0.01) && tl && tl.length > 0) {
      const rawIdx = (playSec / Math.max(0.01, slot.analysis.durationSec)) * (tl.length - 1);
      const idx0 = Math.max(0, Math.min(tl.length - 1, Math.floor(rawIdx)));
      const idx1 = Math.min(tl.length - 1, idx0 + 1);
      const frac = rawIdx - idx0;
      const p0 = tl[idx0];
      const p1 = tl[idx1];

      const interp = (a, b) => {
        if (!isFinite(a)) return b;
        if (!isFinite(b)) return a;
        return a + (b - a) * frac;
      };

      return {
        ...summary,
        momentary: interp(p0.momentary, p1.momentary),
        shortTerm: interp(p0.shortTerm, p1.shortTerm),
        truePeakL: interp(p0.truePeakL, p1.truePeakL),
        truePeakR: interp(p0.truePeakR, p1.truePeakR),
        psr: interp(p0.psr, p1.psr),
        correlation: monoCheckEnabled ? 1.0 : interp(p0.correlation ?? 1, p1.correlation ?? 1),
      };
    }
    return summary;
  }
  return engine.getSnapshot();
}

function formatYouleanValue(val, unit) {
  if (!isFinite(val) || val < -90) return '-';
  return `${val.toFixed(1)} <span class="ylm-unit">${unit}</span>`;
}

function updateReadouts(snap, now = performance.now()) {
  const preset = STREAMING_PRESETS[currentPresetKey];

  // 1. 60fps Stereo Phase Correlation Bar, Thumb & Min-Hold Tick
  const corr = Math.max(-1, Math.min(1, isFinite(ballistics.correlation) ? ballistics.correlation : 1.0));
  const pct = ((corr + 1) / 2) * 100;
  if (phaseThumb) {
    phaseThumb.style.left = `${Math.max(4, Math.min(96, pct))}%`;
  }
  if (phaseFill) {
    if (corr >= 0) {
      phaseFill.style.left = '50%';
      phaseFill.style.width = `${corr * 50}%`;
      phaseFill.style.background = corr > 0.35 ? '#7ac174' : '#5887da';
    } else {
      const w = Math.abs(corr) * 50;
      phaseFill.style.left = `${50 - w}%`;
      phaseFill.style.width = `${w}%`;
      phaseFill.style.background = corr < -0.2 ? '#d95852' : '#d8a648';
    }
  }
  if (phaseMinTick) {
    const minC = Math.max(-1, Math.min(1, isFinite(snap.minCorrelation) ? snap.minCorrelation : 1.0));
    phaseMinTick.style.left = `${Math.max(2, Math.min(98, ((minC + 1) / 2) * 100))}%`;
  }

  // 2. Throttle numeric text readouts to ~5.5 Hz (every 180ms) so digits are calm & readable
  if (now - lastReadoutUpdateTime < 180) {
    return;
  }
  lastReadoutUpdateTime = now;

  const dispST = ballistics.shortTerm > -57.2 ? ballistics.shortTerm : snap.shortTerm;
  const dispInt = isFinite(ballistics.integrated) ? ballistics.integrated : snap.integrated;
  const dispLRA = ballistics.lra;
  const dispPSR = isFinite(ballistics.psr) ? ballistics.psr : snap.psr;
  const dispPLR = isFinite(ballistics.plr) ? ballistics.plr : snap.plr;

  valShortTerm.innerHTML = formatYouleanValue(dispST, 'LUFS');
  valIntegrated.innerHTML = formatYouleanValue(dispInt, 'LUFS');
  valLRA.innerHTML = `${dispLRA.toFixed(1)} <span class="ylm-unit">LU</span>`;
  valPSR.innerHTML = isFinite(dispPSR) && dispPSR > 0 ? `${dispPSR.toFixed(1)} <span class="ylm-unit">LU</span>` : '-';
  valPLR.innerHTML = isFinite(dispPLR) && dispPLR > 0 ? `${dispPLR.toFixed(1)} <span class="ylm-unit">LU</span>` : '-';

  // 3 Rounded Pills
  valMomentaryMax.innerHTML = formatYouleanValue(snap.momentaryMax, 'LUFS');
  valShortTermMax.innerHTML = formatYouleanValue(snap.shortTermMax, 'LUFS');
  valTruePeakMax.innerHTML = formatYouleanValue(snap.maxTruePeak, 'dB');

  if (snap.maxTruePeak > preset.maxTruePeak) {
    valTruePeakMax.style.color = snap.maxTruePeak > 0 ? '#ff6b6b' : '#f6ad55';
  } else {
    valTruePeakMax.style.color = '#ffffff';
  }

  if (valPhaseCorr) {
    const sign = corr >= 0 ? '+' : '';
    valPhaseCorr.textContent = monoCheckEnabled
      ? 'MONO +1.00'
      : `PHASE ${sign}${corr.toFixed(2)}`;
    valPhaseCorr.style.color = corr < -0.15 ? '#ff6b6b' : corr < 0.2 ? '#f6ad55' : '#b8c2d1';
  }

  // Streaming Normalization Gain & A/B Delta Info
  if (isFinite(snap.integrated) && snap.integrated > -70) {
    const gainDB = computeNormalizationGainDB(snap);
    const sign = gainDB > 0 ? '+' : '';
    btnAutoScale.textContent = normPreviewEnabled ? `ON (${sign}${gainDB.toFixed(1)}dB)` : `${sign}${gainDB.toFixed(1)} dB`;
    btnNormPreview.textContent = normPreviewEnabled
      ? `NORM ON (${sign}${gainDB.toFixed(1)} dB)`
      : `NORM PREVIEW (${sign}${gainDB.toFixed(1)} dB)`;

    const otherSlot = getOtherSlot();
    const otherInt = otherSlot.analysis?.summary?.integrated;
    const abText =
      isFinite(otherInt) && otherInt > -70
        ? ` | A/B 차이: ${(snap.integrated - otherInt > 0 ? '+' : '') + (snap.integrated - otherInt).toFixed(1)} LU`
        : '';

    verdictText.title = `${preset.name} (${preset.targetLUFS} LUFS): 노멀라이제이션 게인 ${sign}${gainDB.toFixed(1)} dB | True Peak Max: ${formatDB(snap.maxTruePeak, 1)} dBTP${abText}`;
  } else {
    btnAutoScale.textContent = 'AUTO';
    btnNormPreview.textContent = normPreviewEnabled ? 'NORM ON' : 'NORM PREVIEW';
  }
}

// ================= COLUMN 1: LEFT BAR METER RENDERER =================
function renderBarMeters() {
  const dpr = window.devicePixelRatio || 1;
  const rect = barMeterCanvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;

  const width = Math.floor(rect.width * dpr);
  const height = Math.floor(rect.height * dpr);
  if (barMeterCanvas.width !== width || barMeterCanvas.height !== height) {
    barMeterCanvas.width = width;
    barMeterCanvas.height = height;
  }

  const ctx = barMeterCanvas.getContext('2d');
  ctx.save();
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const topPad = 14;
  const botPad = 48;
  const barH = Math.max(20, H - topPad - botPad);

  const minLUFS = -58;
  const maxLUFS = 0;
  const lufsToY = (val) => {
    const clamped = Math.max(minLUFS, Math.min(maxLUFS, isFinite(val) ? val : minLUFS));
    const norm = (clamped - minLUFS) / (maxLUFS - minLUFS);
    return topPad + barH * (1 - norm);
  };

  const preset = STREAMING_PRESETS[currentPresetKey];
  const targetInt = Math.round(preset.targetLUFS);

  const baseMarks = [0, -3, -6, -9, -18, -23, -27, -36, -45, -54];
  if (!baseMarks.includes(targetInt) && targetInt < 0 && targetInt > -54) {
    baseMarks.push(targetInt);
    baseMarks.sort((a, b) => b - a);
  }

  ctx.font = '600 12.5px "Open Sans", sans-serif';
  ctx.textBaseline = 'middle';

  const leftTickX = 32;
  const rightTickX = W - 18;
  const barCenterX = (leftTickX + 8 + rightTickX) / 2;
  const barW = 24;
  const barX = barCenterX - barW / 2;

  for (const m of baseMarks) {
    const y = lufsToY(m);
    const isTarget = m === targetInt;

    ctx.fillStyle = isTarget ? '#d8a648' : '#e5e9f0';
    ctx.textAlign = 'right';
    ctx.fillText(`${m}`, leftTickX - 4, y);

    ctx.strokeStyle = isTarget ? '#d8a648' : '#9aa3b2';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(leftTickX, y);
    ctx.lineTo(leftTickX + 7, y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(rightTickX - 7, y);
    ctx.lineTo(rightTickX, y);
    ctx.stroke();
  }

  // Bar Fill (Momentary 'M' or Short-Term 'S' mode)
  const palette = THEME_PALETTES[colorTheme] || THEME_PALETTES.classic;
  const activeBarVal = barMeterMode === 'S' ? ballistics.shortTerm : ballistics.momentary;
  const barVal = isFinite(activeBarVal) ? activeBarVal : minLUFS;
  if (barVal > minLUFS + 0.1) {
    const barY = lufsToY(barVal);
    const targetY = lufsToY(preset.targetLUFS);
    const bottomY = topPad + barH;

    const blueTopY = Math.max(barY, targetY);
    if (bottomY > blueTopY) {
      ctx.fillStyle = palette.barMain;
      ctx.fillRect(barX, blueTopY, barW, bottomY - blueTopY);
    }

    if (barY < targetY) {
      ctx.fillStyle = palette.barOver;
      ctx.fillRect(barX, barY, barW, targetY - barY);
    }
  }

  // LRA Bracket [   ]
  const centerRef =
    isFinite(ballistics.integrated) && ballistics.integrated > -54 ? ballistics.integrated : -31;
  const halfLRA = Math.max(2, (ballistics.lra || 4) / 2);
  const brkTopY = lufsToY(centerRef + halfLRA);
  const brkBotY = lufsToY(centerRef - halfLRA);

  ctx.strokeStyle = '#9aa3b2';
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(barX + 3, brkTopY);
  ctx.lineTo(barX - 1, brkTopY);
  ctx.lineTo(barX - 1, brkBotY);
  ctx.lineTo(barX + 3, brkBotY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(barX + barW - 3, brkTopY);
  ctx.lineTo(barX + barW + 1, brkTopY);
  ctx.lineTo(barX + barW + 1, brkBotY);
  ctx.lineTo(barX + barW - 3, brkBotY);
  ctx.stroke();

  // White Triangle Pointer (◀)
  const ptrVal = barVal > minLUFS + 0.2 ? barVal : minLUFS + 2;
  const ptrY = lufsToY(ptrVal);
  ctx.fillStyle = '#e5e9f0';
  ctx.beginPath();
  ctx.moveTo(rightTickX - 8, ptrY);
  ctx.lineTo(rightTickX + 4, ptrY - 6);
  ctx.lineTo(rightTickX + 4, ptrY + 6);
  ctx.closePath();
  ctx.fill();

  // Bottom 'M' / 'S' and 'LUFS' labels
  const footerLineY = topPad + barH + 10;
  ctx.strokeStyle = '#626875';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(barCenterX - 14, footerLineY);
  ctx.lineTo(barCenterX + 14, footerLineY);
  ctx.stroke();

  ctx.fillStyle = '#d8dee9';
  ctx.textAlign = 'center';
  ctx.font = '700 9px "Open Sans", sans-serif';
  ctx.fillText(barMeterMode, barCenterX, footerLineY + 9);

  ctx.font = '600 14px "Open Sans", sans-serif';
  ctx.fillStyle = '#ffffff';
  ctx.fillText('LUFS', barCenterX, footerLineY + 25);

  ctx.restore();
}

// ================= COLUMN 3 & 4: HISTOGRAM + TIME GRAPH + MINIMAP RENDERER =================
function renderHistoryGraph() {
  const dpr = window.devicePixelRatio || 1;
  const rect = historyCanvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;

  const width = Math.floor(rect.width * dpr);
  const height = Math.floor(rect.height * dpr);
  if (historyCanvas.width !== width || historyCanvas.height !== height) {
    historyCanvas.width = width;
    historyCanvas.height = height;
  }

  const ctx = historyCanvas.getContext('2d');
  ctx.save();
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;

  ctx.fillStyle = '#1e2128';
  ctx.fillRect(0, 0, W, H);

  const topPad = 14;
  const minimapH = 32;
  const timeRowH = 28;
  const botPad = minimapH + timeRowH + 8;
  const plotH = Math.max(30, H - topPad - botPad);

  const histAxisX = 108;
  const plotLeft = 152;
  const plotRight = W - 12;
  const plotW = Math.max(20, plotRight - plotLeft);

  const minLUFS = -58;
  const maxLUFS = 0;
  const lufsToY = (val) => {
    const clamped = Math.max(minLUFS, Math.min(maxLUFS, isFinite(val) ? val : minLUFS));
    const norm = (clamped - minLUFS) / (maxLUFS - minLUFS);
    return topPad + plotH * (1 - norm);
  };

  const preset = STREAMING_PRESETS[currentPresetKey];
  const targetInt = Math.round(preset.targetLUFS);
  const targetY = lufsToY(preset.targetLUFS);

  // 1. Horizontal Grid Lines & Scale Labels
  const baseMarks = [0, -3, -6, -9, -18, -23, -27, -36, -45, -54];
  if (!baseMarks.includes(targetInt) && targetInt < 0 && targetInt > -54) {
    baseMarks.push(targetInt);
    baseMarks.sort((a, b) => b - a);
  }

  ctx.font = '600 12.5px "Open Sans", sans-serif';
  ctx.textBaseline = 'middle';

  for (const m of baseMarks) {
    const y = lufsToY(m);
    const isTarget = m === targetInt;

    if (layers.G) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(6, y);
      ctx.lineTo(histAxisX, y);
      ctx.stroke();

      ctx.strokeStyle = isTarget ? 'rgba(216, 166, 72, 0.25)' : 'rgba(255, 255, 255, 0.055)';
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotRight, y);
      ctx.stroke();
    }

    ctx.fillStyle = isTarget ? '#d8a648' : '#e5e9f0';
    ctx.textAlign = 'right';
    ctx.fillText(`${m}`, plotLeft - 14, y);

    ctx.strokeStyle = isTarget ? '#d8a648' : '#9aa3b2';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(plotLeft - 9, y);
    ctx.lineTo(plotLeft - 2, y);
    ctx.stroke();
  }

  // 2. Render Loudness Distribution Histogram
  const slot = getSlot();
  const isFileMode = currentMode === 'file' && Boolean(slot.analysis);
  const activeHist = isFileMode ? slot.histogram : liveHistogram;
  let maxBinCount = 0;
  for (let i = 0; i < HIST_BINS; i++) {
    if (activeHist[i] > maxBinCount) maxBinCount = activeHist[i];
  }

  if (maxBinCount > 0) {
    const maxHistWidth = histAxisX - 8;
    ctx.beginPath();
    ctx.moveTo(histAxisX, topPad + plotH);

    for (let i = 0; i < HIST_BINS; i++) {
      const binLUFS = -60 + ((i + 0.5) / HIST_BINS) * 60;
      if (binLUFS < minLUFS) continue;
      const y = lufsToY(binLUFS);
      const w = (activeHist[i] / maxBinCount) * maxHistWidth;
      ctx.lineTo(histAxisX - w, y);
    }

    ctx.lineTo(histAxisX, topPad);
    ctx.closePath();

    ctx.fillStyle = 'rgba(88, 135, 218, 0.38)';
    ctx.fill();
    ctx.strokeStyle = '#6b9be8';
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }

  // Vertical Histogram Axis Line (Red above target, Blue below target)
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#d95852';
  ctx.beginPath();
  ctx.moveTo(histAxisX, topPad);
  ctx.lineTo(histAxisX, targetY);
  ctx.stroke();

  ctx.strokeStyle = '#5887da';
  ctx.beginPath();
  ctx.moveTo(histAxisX, targetY);
  ctx.lineTo(histAxisX, topPad + plotH);
  ctx.stroke();

  // 3. Render Main Time-Series Loudness Graph
  const sourceTimeline = isFileMode ? slot.analysis.timeline : liveHistory;

  const lastTime = sourceTimeline.length > 0 ? sourceTimeline[sourceTimeline.length - 1].timeSec : 0;
  const totalDuration = isFileMode
    ? slot.analysis.durationSec
    : Math.max(timeWindowSec || 30, lastTime);

  let startTime = 0;
  let endTime = totalDuration;

  if (timeWindowSec > 0) {
    if (isFileMode) {
      const playhead = getFilePlayheadSec();
      endTime = Math.min(totalDuration, Math.max(timeWindowSec, playhead + timeWindowSec * 0.15));
      startTime = Math.max(0, endTime - timeWindowSec);
    } else {
      endTime = Math.max(timeWindowSec, lastTime);
      startTime = endTime - timeWindowSec;
    }
  }

  const spanSec = Math.max(0.5, endTime - startTime);
  const timeToX = (t) => plotLeft + ((t - startTime) / spanSec) * plotW;

  const miniTop = H - minimapH - 4;
  const miniLeft = plotLeft - 36;
  const miniW = plotRight - miniLeft;

  // Save geometry for accurate click/drag seeking
  lastGraphView = { startTime, endTime, totalDuration, plotLeft, plotRight, miniLeft, miniW, miniTop, topPad, plotH };

  // Draw Other Slot (A/B Comparison) Ghost Curve if Layer D is active & other slot has data
  const otherSlot = getOtherSlot();
  if (layers.D && otherSlot.analysis && otherSlot.analysis.timeline.length > 1) {
    const otherVisible = otherSlot.analysis.timeline.filter(
      (pt) => pt.timeSec >= startTime && pt.timeSec <= endTime
    );
    if (otherVisible.length > 1) {
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < otherVisible.length; i++) {
        const x = timeToX(otherVisible[i].timeSec);
        const y = lufsToY(otherVisible[i].shortTerm);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(216, 166, 72, 0.75)';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.restore();
    }
  }

  if (sourceTimeline.length > 1) {
    const visible = sourceTimeline.filter((pt) => pt.timeSec >= startTime && pt.timeSec <= endTime);

    if (visible.length > 1) {
      const getVal = (pt) => (layers.M && !layers.S ? pt.momentary : pt.shortTerm);

      const drawCurvePath = () => {
        ctx.beginPath();
        ctx.moveTo(timeToX(visible[0].timeSec), topPad + plotH);
        for (let i = 0; i < visible.length; i++) {
          ctx.lineTo(timeToX(visible[i].timeSec), lufsToY(getVal(visible[i])));
        }
        ctx.lineTo(timeToX(visible[visible.length - 1].timeSec), topPad + plotH);
        ctx.closePath();
      };

      // Below Target Clip -> Theme Main Fill & Stroke
      const palette = THEME_PALETTES[colorTheme] || THEME_PALETTES.classic;
      ctx.save();
      ctx.beginPath();
      ctx.rect(plotLeft, targetY, plotW, topPad + plotH - targetY);
      ctx.clip();
      drawCurvePath();
      ctx.fillStyle = palette.curveFill;
      ctx.fill();
      ctx.strokeStyle = palette.curveStroke;
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.restore();

      // Above Target Clip -> Theme Over Fill & Stroke
      ctx.save();
      ctx.beginPath();
      ctx.rect(plotLeft, topPad, plotW, Math.max(0, targetY - topPad));
      ctx.clip();
      drawCurvePath();
      ctx.fillStyle = palette.overFill;
      ctx.fill();
      ctx.strokeStyle = palette.overStroke;
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.restore();

      // Secondary Momentary Line if both S and M are active
      if (layers.S && layers.M) {
        ctx.beginPath();
        for (let i = 0; i < visible.length; i++) {
          const x = timeToX(visible[i].timeSec);
          const y = lufsToY(visible[i].momentary);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.38)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Dynamics / Stereo Phase Trace if Layer D is active
      if (layers.D) {
        ctx.beginPath();
        for (let i = 0; i < visible.length; i++) {
          const x = timeToX(visible[i].timeSec);
          // Map correlation [-1..+1] to bottom strip of plot (-54 to -36 LUFS)
          const corr = visible[i].correlation ?? 1;
          const mappedLUFS = -45 + corr * 8;
          const y = lufsToY(mappedLUFS);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(122, 193, 116, 0.75)';
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }

      // True Peak Warning Spikes along top if Layer T is enabled
      if (layers.T) {
        for (let i = 0; i < visible.length; i++) {
          const pt = visible[i];
          const maxTP = Math.max(pt.truePeakL, pt.truePeakR);
          if (maxTP > preset.maxTruePeak) {
            const x = timeToX(pt.timeSec);
            ctx.fillStyle = maxTP > 0 ? '#ef4444' : '#d8a648';
            ctx.fillRect(x - 1, topPad, 2.5, 8);
          }
        }
      }

      // File Playhead Vertical Line
      if (isFileMode) {
        const playSec = getFilePlayheadSec();
        if (playSec >= startTime && playSec <= endTime) {
          const px = timeToX(playSec);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(px, topPad);
          ctx.lineTo(px, topPad + plotH);
          ctx.stroke();
        }
      }
    }
  }

  // 4. Time Axis Labels
  const timeLabelY = topPad + plotH + 16;
  ctx.fillStyle = '#e5e9f0';
  ctx.font = '600 12.5px "Open Sans", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const numTicks = 5;
  for (let i = 0; i <= numTicks; i++) {
    const frac = i / numTicks;
    const sec = startTime + frac * spanSec;
    const tx = plotLeft + frac * plotW;
    const label = sec >= 60 ? formatTime(sec) : `${Math.round(sec)}s`;
    ctx.fillText(label, tx, timeLabelY);
  }

  // 5. Bottom Minimap Timeline Strip
  ctx.strokeStyle = '#5c6270';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(miniLeft + 4, miniTop + 2);
  ctx.lineTo(miniLeft, miniTop + 2);
  ctx.lineTo(miniLeft, miniTop + minimapH - 2);
  ctx.lineTo(miniLeft + 4, miniTop + minimapH - 2);
  ctx.stroke();

  const innerMiniX = miniLeft + 8;
  const innerMiniW = Math.max(10, miniW - 16);

  if (sourceTimeline.length > 1) {
    const fullDur = Math.max(0.5, totalDuration);
    ctx.beginPath();
    ctx.moveTo(innerMiniX, miniTop + minimapH - 2);
    const step = Math.max(1, Math.floor(sourceTimeline.length / 220));
    for (let i = 0; i < sourceTimeline.length; i += step) {
      const pt = sourceTimeline[i];
      const mx = innerMiniX + (pt.timeSec / fullDur) * innerMiniW;
      const val = isFinite(pt.shortTerm) ? pt.shortTerm : pt.momentary;
      const norm = Math.max(0, Math.min(1, (val - minLUFS) / (maxLUFS - minLUFS)));
      const my = miniTop + minimapH - 2 - norm * (minimapH - 6);
      ctx.lineTo(mx, my);
    }
    const lastPt = sourceTimeline[sourceTimeline.length - 1];
    ctx.lineTo(innerMiniX + (lastPt.timeSec / fullDur) * innerMiniW, miniTop + minimapH - 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(88, 135, 218, 0.55)';
    ctx.fill();
    ctx.strokeStyle = '#6b9be8';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Dynamic Viewport Box in Minimap
  const fullDur = Math.max(0.5, totalDuration);
  const vpStartRatio = Math.max(0, Math.min(1, startTime / fullDur));
  const vpEndRatio = Math.max(vpStartRatio + 0.04, Math.min(1, endTime / fullDur));
  const vpBoxX = innerMiniX + vpStartRatio * innerMiniW;
  const vpBoxW = Math.max(24, (vpEndRatio - vpStartRatio) * innerMiniW);

  ctx.fillStyle = 'rgba(70, 75, 87, 0.55)';
  ctx.strokeStyle = '#8b92a0';
  ctx.lineWidth = 1.5;
  ctx.fillRect(vpBoxX, miniTop, vpBoxW, minimapH - 2);
  ctx.strokeRect(vpBoxX, miniTop, vpBoxW, minimapH - 2);

  // Minimap Playhead Line
  if (isFileMode) {
    const playRatio = Math.max(0, Math.min(1, getFilePlayheadSec() / fullDur));
    const mpx = innerMiniX + playRatio * innerMiniW;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(mpx, miniTop);
    ctx.lineTo(mpx, miniTop + minimapH - 2);
    ctx.stroke();
  }

  // Hover Crosshair Readout
  if (
    hoverPoint &&
    hoverPoint.x >= plotLeft &&
    hoverPoint.x <= plotRight &&
    hoverPoint.y >= topPad &&
    hoverPoint.y <= topPad + plotH &&
    sourceTimeline.length > 1
  ) {
    const hoverRatio = (hoverPoint.x - plotLeft) / plotW;
    const hoverSec = startTime + hoverRatio * spanSec;
    let closest = sourceTimeline[0];
    let minDiff = Infinity;
    for (let i = 0; i < sourceTimeline.length; i++) {
      const d = Math.abs(sourceTimeline[i].timeSec - hoverSec);
      if (d < minDiff) {
        minDiff = d;
        closest = sourceTimeline[i];
      }
    }

    const cx = timeToX(closest.timeSec);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, topPad);
    ctx.lineTo(cx, topPad + plotH);
    ctx.stroke();

    const tpPeak = Math.max(closest.truePeakL, closest.truePeakR);
    const ptCorr = closest.correlation ?? 1;
    const labelText = `${formatTime(closest.timeSec)} | S: ${formatLUFS(closest.shortTerm, 1)} | M: ${formatLUFS(closest.momentary, 1)} | TP: ${formatDB(tpPeak, 1)} dB | Phase: ${(ptCorr >= 0 ? '+' : '') + ptCorr.toFixed(2)}`;
    ctx.font = '700 11.5px "Open Sans", sans-serif';
    const textW = ctx.measureText(labelText).width + 18;
    const bx = Math.min(plotRight - textW - 4, Math.max(plotLeft + 4, cx - textW / 2));
    const by = topPad + 6;

    ctx.fillStyle = 'rgba(28, 31, 38, 0.94)';
    ctx.strokeStyle = '#5887da';
    ctx.lineWidth = 1;
    ctx.fillRect(bx, by, textW, 24);
    ctx.strokeRect(bx, by, textW, 24);

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, bx + 9, by + 12);
  }

  ctx.restore();
}

// ================= PRO EXPORT SUITE: PNG IMAGE / CSV / JSON =================
function exportPNGImage() {
  const snap = getActiveSnapshot();
  const preset = STREAMING_PRESETS[currentPresetKey];
  const slot = getSlot();
  const trackName = slot.fileName || (currentMode === 'live' ? 'Live Audio Session' : 'Audio Measurement');

  const W = 1600;
  const H = 900;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#1e2128';
  ctx.fillRect(0, 0, W, H);

  ctx.fillStyle = '#e2e3e6';
  ctx.fillRect(0, 0, W, 56);
  ctx.fillRect(0, H - 48, W, 48);

  ctx.fillStyle = '#1c1e24';
  ctx.font = '800 24px "Open Sans", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('Vibe Loudness Meter — Loudness Compliance Report', W / 2, 28);

  ctx.font = '700 14px "Open Sans", sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`FILE: ${trackName} (Slot ${activeSlotKey})`, 20, 28);
  ctx.textAlign = 'right';
  ctx.fillText(`TARGET: ${preset.name} (${preset.targetLUFS} LUFS / ${preset.maxTruePeak} dBTP)`, W - 20, 28);

  ctx.drawImage(barMeterCanvas, 16, 68, 150, H - 130);

  const rx = 182;
  const rw = 240;
  ctx.fillStyle = '#363c48';
  ctx.fillRect(rx, 74, rw, 390);
  ctx.fillStyle = '#626f82';
  ctx.fillRect(rx, 74 + 78, rw, 78);

  const stackRows = [
    { val: `${formatLUFS(snap.shortTerm, 1)} LUFS`, lbl: 'SHORT TERM' },
    { val: `${formatLUFS(snap.integrated, 1)} LUFS`, lbl: 'INTEGRATED' },
    { val: `${snap.lra.toFixed(1)} LU`, lbl: 'LOUDNESS RANGE' },
    { val: isFinite(snap.psr) && snap.psr > 0 ? `${snap.psr.toFixed(1)} LU` : '-', lbl: 'DYNAMICS (PSR)' },
    { val: isFinite(snap.plr) && snap.plr > 0 ? `${snap.plr.toFixed(1)} LU` : '-', lbl: 'AVG. DYNAMICS (PLR)' },
  ];

  stackRows.forEach((r, idx) => {
    const cy = 74 + idx * 78 + 39;
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 28px "Open Sans", sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(r.val, rx + rw / 2, cy - 8);

    ctx.fillStyle = '#d8dee9';
    ctx.font = '600 13px "Open Sans", sans-serif';
    ctx.fillText(r.lbl, rx + rw / 2, cy + 20);
  });

  const pills = [
    { val: `${formatLUFS(snap.momentaryMax, 1)} LUFS`, lbl: 'MOMENTARY MAX' },
    { val: `${formatLUFS(snap.shortTermMax, 1)} LUFS`, lbl: 'SHORT TERM MAX' },
    { val: `${formatDB(snap.maxTruePeak, 1)} dBTP`, lbl: 'TRUE PEAK MAX' },
  ];

  pills.forEach((p, idx) => {
    const py = 490 + idx * 82;
    ctx.fillStyle = '#363c48';
    ctx.beginPath();
    ctx.roundRect(rx + 10, py, rw - 20, 46, 23);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = '700 24px "Open Sans", sans-serif';
    ctx.fillText(p.val, rx + rw / 2, py + 24);

    ctx.fillStyle = '#d0d6e0';
    ctx.font = '600 12px "Open Sans", sans-serif';
    ctx.fillText(p.lbl, rx + rw / 2, py + 62);
  });

  ctx.drawImage(historyCanvas, 440, 68, W - 456, H - 130);

  const gainDB = computeNormalizationGainDB(snap);
  const sign = gainDB > 0 ? '+' : '';
  ctx.fillStyle = '#1c1e24';
  ctx.font = '700 14px "Open Sans", sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(
    `Normalization Gain (${preset.name}): ${sign}${gainDB.toFixed(1)} dB   |   Phase Correlation: ${(snap.correlation ?? 1).toFixed(2)} (Min: ${(snap.minCorrelation ?? 1).toFixed(2)})   |   True Peak Clips: ${snap.clipCount}`,
    20,
    H - 24
  );
  ctx.textAlign = 'right';
  ctx.fillText(`Generated: ${new Date().toLocaleString()}`, W - 20, H - 24);

  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vibe-loudness-report-${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

function exportCSVTimeline() {
  const slot = getSlot();
  const isFileMode = currentMode === 'file' && Boolean(slot.analysis);
  const timeline = isFileMode ? slot.analysis.timeline : liveHistory;

  if (!timeline || timeline.length === 0) {
    alert('내보낼 타임라인 측정 데이터가 없습니다.');
    return;
  }

  const lines = ['Time_Sec,Momentary_LUFS,ShortTerm_LUFS,Integrated_LUFS,TruePeak_L_dBTP,TruePeak_R_dBTP,Phase_Correlation'];
  for (const pt of timeline) {
    lines.push(
      [
        pt.timeSec.toFixed(2),
        isFinite(pt.momentary) ? pt.momentary.toFixed(2) : '',
        isFinite(pt.shortTerm) ? pt.shortTerm.toFixed(2) : '',
        isFinite(pt.integrated) ? pt.integrated.toFixed(2) : '',
        isFinite(pt.truePeakL) ? pt.truePeakL.toFixed(2) : '',
        isFinite(pt.truePeakR) ? pt.truePeakR.toFixed(2) : '',
        isFinite(pt.correlation) ? pt.correlation.toFixed(3) : '1.000',
      ].join(',')
    );
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `vibe-loudness-timeline-${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportLoudnessReport() {
  const snap = getActiveSnapshot();
  const preset = STREAMING_PRESETS[currentPresetKey];
  const slot = getSlot();
  const report = {
    generatedAt: new Date().toISOString(),
    application: 'Vibe Loudness Meter',
    standard: 'ITU-R BS.1770-4 / EBU R128',
    activeSlot: activeSlotKey,
    sourceMode: currentMode,
    fileName: slot.fileName || null,
    targetPreset: preset,
    normalizationGainDB: Number(computeNormalizationGainDB(snap).toFixed(2)),
    metrics: {
      integratedLUFS: isFinite(snap.integrated) ? Number(snap.integrated.toFixed(2)) : null,
      shortTermMaxLUFS: isFinite(snap.shortTermMax) ? Number(snap.shortTermMax.toFixed(2)) : null,
      momentaryMaxLUFS: isFinite(snap.momentaryMax) ? Number(snap.momentaryMax.toFixed(2)) : null,
      loudnessRangeLU: Number(snap.lra.toFixed(2)),
      truePeakMaxDBTP: isFinite(snap.maxTruePeak) ? Number(snap.maxTruePeak.toFixed(2)) : null,
      plrLU: isFinite(snap.plr) ? Number(snap.plr.toFixed(2)) : null,
      psrLU: isFinite(snap.psr) ? Number(snap.psr.toFixed(2)) : null,
      phaseCorrelation: Number((snap.correlation ?? 1).toFixed(2)),
      minPhaseCorrelation: Number((snap.minCorrelation ?? 1).toFixed(2)),
      truePeakClips: snap.clipCount,
      durationSec: Number(snap.durationSec.toFixed(2)),
    },
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `vibe-loudness-report-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ================= ALWAYS-ON-TOP (DOCUMENT PICTURE-IN-PICTURE) =================
async function toggleAlwaysOnTopPiP() {
  if ('documentPictureInPicture' in window) {
    try {
      if (window.documentPictureInPicture.window) {
        window.documentPictureInPicture.window.close();
        return;
      }

      const pipWin = await window.documentPictureInPicture.requestWindow({
        width: 320,
        height: 560,
      });

      [...document.styleSheets].forEach((styleSheet) => {
        try {
          const cssRules = [...styleSheet.cssRules].map((r) => r.cssText).join('');
          const style = document.createElement('style');
          style.textContent = cssRules;
          pipWin.document.head.appendChild(style);
        } catch (_) {
          if (styleSheet.href) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = styleSheet.href;
            pipWin.document.head.appendChild(link);
          }
        }
      });

      pipWin.document.body.classList.add('mini-mode');
      const appShell = document.getElementById('app');
      pipWin.document.body.appendChild(appShell);

      pipWin.addEventListener('pagehide', () => {
        document.body.appendChild(appShell);
      });
      return;
    } catch (err) {
      console.warn('Document PiP fallback to mini mode:', err);
    }
  }

  document.body.classList.toggle('mini-mode');
  btnMiniMode.classList.toggle('active', document.body.classList.contains('mini-mode'));
}

// ================= HELPERS & EVENT LISTENERS =================
function formatTime(sec) {
  if (!isFinite(sec) || sec < 0) return '0s';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  if (m === 0) return `${s}s`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function resetMeters() {
  engine.reset();
  ballistics.reset();
  lastReadoutUpdateTime = 0;
  liveHistory = [];
  liveHistogram.fill(0);
  const slot = getSlot();
  if (currentMode === 'file' && !fileIsPlaying) {
    slot.analysis = null;
    slot.histogram.fill(0);
    slot.audioBuffer = null;
    slot.fileName = '';
    slot.pauseOffsetSec = 0;
    updateABButtonsUI();
    setStatus('standby', '');
    fileMetaInfo.textContent = `SLOT ${activeSlotKey}: READY`;
  }
}

function toggleNormPreview() {
  normPreviewEnabled = !normPreviewEnabled;
  lastReadoutUpdateTime = 0;
  btnNormPreview.classList.toggle('active', normPreviewEnabled);
  if (btnFloatAudio) btnFloatAudio.classList.toggle('active-red', normPreviewEnabled);
  applyRealtimePlaybackRouting();
}

function toggleMonoCheck() {
  monoCheckEnabled = !monoCheckEnabled;
  lastReadoutUpdateTime = 0;
  btnMonoCheck.classList.toggle('active', monoCheckEnabled);
  if (fileIsPlaying) {
    const curSec = getFilePlayheadSec();
    startFilePlayback(curSec);
  }
}

function syncLayerButtonsUI() {
  document.querySelectorAll('.ylm-circle-toggle').forEach((btn) => {
    const key = btn.dataset.layer;
    if (key && key in layers) {
      btn.classList.toggle('active', layers[key]);
    }
  });
  if (btnFloatAudio) btnFloatAudio.classList.toggle('active-red', normPreviewEnabled);
  if (btnFloatPeaks) btnFloatPeaks.classList.toggle('active-red', layers.T);
  if (btnFloatTarget) btnFloatTarget.classList.toggle('active-red', layers.G);
}

btnLiveInput.addEventListener('click', startLiveDeviceInput);
deviceSelect.addEventListener('change', () => {
  if (currentMode === 'live') {
    currentMode = 'standby';
    startLiveDeviceInput();
  }
});
btnSystemAudio.addEventListener('click', startSystemAudioCapture);
btnTestTone.addEventListener('click', toggleReferenceTone);

btnNormPreview.addEventListener('click', toggleNormPreview);
btnAutoScale.addEventListener('click', toggleNormPreview);
btnMonoCheck.addEventListener('click', toggleMonoCheck);

if (btnIntLra) {
  btnIntLra.addEventListener('click', () => {
    barMeterMode = barMeterMode === 'M' ? 'S' : 'M';
    btnIntLra.textContent = barMeterMode === 'M' ? 'INT-LRA' : 'ST-LRA';
  });
}

fileInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) handleFileLoad(file);
  fileInput.value = '';
});

presetSelect.addEventListener('change', (e) => {
  currentPresetKey = e.target.value;
  const p = STREAMING_PRESETS[currentPresetKey];
  if (inpCustomTarget && p) inpCustomTarget.value = p.targetLUFS.toFixed(1);
  if (inpCustomCeiling && p) inpCustomCeiling.value = p.maxTruePeak.toFixed(1);
  lastReadoutUpdateTime = 0;
  applyRealtimePlaybackRouting();
});

btnReset.addEventListener('click', resetMeters);

btnMiniMode.addEventListener('click', () => {
  document.body.classList.toggle('mini-mode');
  btnMiniMode.classList.toggle('active', document.body.classList.contains('mini-mode'));
});

btnPiP.addEventListener('click', toggleAlwaysOnTopPiP);

// Export Menu Listeners
btnExport.addEventListener('click', (e) => {
  e.stopPropagation();
  if (settingsMenu) settingsMenu.classList.add('hidden');
  exportMenu.classList.toggle('hidden');
});
btnExportPNG.addEventListener('click', () => {
  exportMenu.classList.add('hidden');
  exportPNGImage();
});
btnExportCSV.addEventListener('click', () => {
  exportMenu.classList.add('hidden');
  exportCSVTimeline();
});
btnExportJSON.addEventListener('click', () => {
  exportMenu.classList.add('hidden');
  exportLoudnessReport();
});

// Settings Menu Listeners
if (btnSettings && settingsMenu) {
  btnSettings.addEventListener('click', (e) => {
    e.stopPropagation();
    exportMenu.classList.add('hidden');
    settingsMenu.classList.toggle('hidden');
  });
  settingsMenu.addEventListener('click', (e) => {
    e.stopPropagation();
  });
}
if (btnCloseSettings) {
  btnCloseSettings.addEventListener('click', () => {
    settingsMenu.classList.add('hidden');
  });
}
if (inpCustomTarget) {
  inpCustomTarget.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if (isFinite(val) && val >= -48 && val <= 0) {
      STREAMING_PRESETS[currentPresetKey].targetLUFS = val;
      lastReadoutUpdateTime = 0;
      applyRealtimePlaybackRouting();
    }
  });
}
if (inpCustomCeiling) {
  inpCustomCeiling.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    if (isFinite(val) && val >= -12 && val <= 0) {
      STREAMING_PRESETS[currentPresetKey].maxTruePeak = val;
      lastReadoutUpdateTime = 0;
      applyRealtimePlaybackRouting();
    }
  });
}
if (selBallistics) {
  selBallistics.addEventListener('change', (e) => {
    ballisticsSpeed = e.target.value;
  });
}
if (selTheme) {
  selTheme.addEventListener('change', (e) => {
    colorTheme = e.target.value;
  });
}
if (btnClearAllSlots) {
  btnClearAllSlots.addEventListener('click', () => {
    stopAllActiveAudio();
    ['A', 'B'].forEach((k) => {
      slots[k].audioBuffer = null;
      slots[k].fileName = '';
      slots[k].analysis = null;
      slots[k].histogram.fill(0);
      slots[k].pauseOffsetSec = 0;
    });
    resetMeters();
    updateABButtonsUI();
    settingsMenu.classList.add('hidden');
  });
}

window.addEventListener('click', () => {
  if (exportMenu) exportMenu.classList.add('hidden');
  if (settingsMenu) settingsMenu.classList.add('hidden');
});

// A / B Slot Listeners
btnSlotA.addEventListener('click', () => switchABSlot('A'));
btnSlotB.addEventListener('click', () => switchABSlot('B'));

windowSelector.addEventListener('change', (e) => {
  timeWindowSec = Number(e.target.value);
});

btnPauseGraph.addEventListener('click', () => {
  graphPaused = !graphPaused;
  if (lblGraphMode) lblGraphMode.textContent = graphPaused ? 'PAUSED' : 'CONTINUOUS';
});

// Layer Toggles (S, M, T, D, G)
document.querySelectorAll('.ylm-circle-toggle').forEach((btn) => {
  btn.addEventListener('click', () => {
    const key = btn.dataset.layer;
    if (!key) return;
    layers[key] = !layers[key];
    syncLayerButtonsUI();
  });
});

if (btnFloatAudio) {
  btnFloatAudio.addEventListener('click', toggleNormPreview);
}
if (btnFloatPeaks) {
  btnFloatPeaks.addEventListener('click', () => {
    layers.T = !layers.T;
    syncLayerButtonsUI();
  });
}
if (btnFloatTarget) {
  btnFloatTarget.addEventListener('click', () => {
    layers.G = !layers.G;
    syncLayerButtonsUI();
  });
}

btnPlayPause.addEventListener('click', () => {
  const slot = getSlot();
  if (slot.audioBuffer) {
    if (fileIsPlaying) pauseFilePlayback();
    else startFilePlayback(slot.pauseOffsetSec);
  } else {
    startLiveDeviceInput();
  }
});

btnStopFile.addEventListener('click', () => {
  if (getSlot().audioBuffer) stopFilePlayback();
  else stopAllActiveAudio();
});

// Accurate Click & Drag Scrubbing on Main Graph and Bottom Minimap
function seekFromCanvasCoords(clientX) {
  const slot = getSlot();
  if (currentMode !== 'file' || !slot.audioBuffer || !slot.analysis) return;
  const rect = historyCanvas.getBoundingClientRect();
  const x = clientX - rect.left;

  let targetSec = 0;
  if (dragSeekZone === 'minimap') {
    const innerX = lastGraphView.miniLeft + 8;
    const innerW = Math.max(10, lastGraphView.miniW - 16);
    const ratio = Math.max(0, Math.min(1, (x - innerX) / innerW));
    targetSec = ratio * slot.audioBuffer.duration;
  } else {
    const plotW = Math.max(20, lastGraphView.plotRight - lastGraphView.plotLeft);
    const ratio = Math.max(0, Math.min(1, (x - lastGraphView.plotLeft) / plotW));
    const span = Math.max(0.5, lastGraphView.endTime - lastGraphView.startTime);
    targetSec = lastGraphView.startTime + ratio * span;
  }

  targetSec = Math.max(0, Math.min(slot.audioBuffer.duration - 0.05, targetSec));
  slot.pauseOffsetSec = targetSec;
  lastReadoutUpdateTime = 0;
  if (fileIsPlaying) {
    startFilePlayback(targetSec);
  }
}

historyCanvas.addEventListener('mousedown', (e) => {
  const slot = getSlot();
  if (currentMode !== 'file' || !slot.audioBuffer || !slot.analysis) return;
  const rect = historyCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (y >= lastGraphView.miniTop - 6 && x >= lastGraphView.miniLeft) {
    isDraggingSeek = true;
    dragSeekZone = 'minimap';
    seekFromCanvasCoords(e.clientX);
  } else if (x >= lastGraphView.plotLeft && x <= lastGraphView.plotRight) {
    isDraggingSeek = true;
    dragSeekZone = 'main';
    seekFromCanvasCoords(e.clientX);
  }
});

window.addEventListener('mousemove', (e) => {
  if (isDraggingSeek) {
    seekFromCanvasCoords(e.clientX);
  }
});

window.addEventListener('mouseup', () => {
  isDraggingSeek = false;
  dragSeekZone = null;
});

historyCanvas.addEventListener('mousemove', (e) => {
  const rect = historyCanvas.getBoundingClientRect();
  hoverPoint = { x: e.clientX - rect.left, y: e.clientY - rect.top };
});

historyCanvas.addEventListener('mouseleave', () => {
  hoverPoint = null;
});

window.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropOverlay.classList.remove('hidden');
});

window.addEventListener('dragleave', (e) => {
  if (e.relatedTarget === null) {
    dropOverlay.classList.add('hidden');
  }
});

window.addEventListener('drop', (e) => {
  e.preventDefault();
  dropOverlay.classList.add('hidden');
  const file = e.dataTransfer?.files?.[0];
  if (file) handleFileLoad(file);
});

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space' && getSlot().audioBuffer) {
    e.preventDefault();
    btnPlayPause.click();
  } else if (e.key === 'r' || e.key === 'R') {
    resetMeters();
  } else if (e.key === 'm' || e.key === 'M') {
    btnMiniMode.click();
  } else if (e.key === 'n' || e.key === 'N') {
    toggleNormPreview();
  } else if (e.key === 'a' || e.key === 'A') {
    switchABSlot('A');
  } else if (e.key === 'b' || e.key === 'B') {
    switchABSlot('B');
  }
});

// ================= MAIN 60FPS RENDER LOOP =================
function tick(now = performance.now()) {
  const dt = (now - lastTickTime) / 1000;
  lastTickTime = now;

  const snap = getActiveSnapshot();
  updateBallistics(snap, dt);
  updateReadouts(snap, now);
  renderHistoryGraph();
  renderBarMeters();
  requestAnimationFrame(tick);
}

syncLayerButtonsUI();
populateAudioDevices();
requestAnimationFrame(tick);

// ================= PWA SERVICE WORKER & INSTALL PROMPT =================
const btnInstallPWA = document.getElementById('btnInstallPWA');
let deferredInstallPrompt = null;

if (
  'serviceWorker' in navigator &&
  (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')
) {
  window.addEventListener('load', () => {
    navigator.serviceWorker
      .register('./sw.js')
      .then((reg) => reg.update())
      .catch((err) => {
        console.warn('ServiceWorker registration failed:', err);
      });
  });
}

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredInstallPrompt = e;
  if (btnInstallPWA) btnInstallPWA.classList.remove('hidden');
});

if (btnInstallPWA) {
  btnInstallPWA.addEventListener('click', async () => {
    if (!deferredInstallPrompt) return;
    deferredInstallPrompt.prompt();
    const { outcome } = await deferredInstallPrompt.userChoice;
    if (outcome === 'accepted') {
      btnInstallPWA.classList.add('hidden');
    }
    deferredInstallPrompt = null;
  });
}

window.addEventListener('appinstalled', () => {
  if (btnInstallPWA) btnInstallPWA.classList.add('hidden');
  deferredInstallPrompt = null;
});
