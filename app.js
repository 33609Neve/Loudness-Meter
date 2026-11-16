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
let timeWindowSec = 30; // 10, 30, 60, 300, or 0 (full)
let graphPaused = false;

// Graph layer toggles (S, M, T, D, G)
const layers = {
  S: true,  // Short-Term curve
  M: false, // Momentary curve
  T: true,  // True Peak markers
  D: false, // Dynamics PSR overlay
  G: true,  // Grid & Target line
};

// Live history buffer (10 Hz points) & Histogram bins (-60 LUFS to 0 LUFS, 120 bins of 0.5 LU)
let liveHistory = [];
const HIST_BINS = 120;
let liveHistogram = new Float64Array(HIST_BINS);

// Active WebAudio nodes
let activeStream = null;
let sourceNode = null;
let processorNode = null;
let zeroGainNode = null;
let toneOscNode = null;
let toneGainNode = null;

// File analysis & playback state
let loadedAudioBuffer = null;
let loadedFileName = '';
let offlineAnalysisResult = null;
let offlineHistogram = new Float64Array(HIST_BINS);
let fileSourceNode = null;
let fileIsPlaying = false;
let filePlayStartCtxTime = 0;
let filePauseOffsetSec = 0;

// Graph hover state
let hoverPoint = null;

// ================= DOM ELEMENTS =================
const elStatusText = document.getElementById('statusText');
const btnLiveInput = document.getElementById('btnLiveInput');
const deviceSelect = document.getElementById('deviceSelect');
const btnSystemAudio = document.getElementById('btnSystemAudio');
const fileInput = document.getElementById('fileInput');
const btnTestTone = document.getElementById('btnTestTone');

const presetSelect = document.getElementById('presetSelect');
const btnReset = document.getElementById('btnReset');
const btnMiniMode = document.getElementById('btnMiniMode');
const btnPiP = document.getElementById('btnPiP');
const btnExport = document.getElementById('btnExport');
const btnAutoScale = document.getElementById('btnAutoScale');

// Readout DOM (Youlean 5-row stack + 3 pills)
const valShortTerm = document.getElementById('valShortTerm');
const valIntegrated = document.getElementById('valIntegrated');
const valLRA = document.getElementById('valLRA');
const valPSR = document.getElementById('valPSR');
const valPLR = document.getElementById('valPLR');

const valMomentaryMax = document.getElementById('valMomentaryMax');
const valShortTermMax = document.getElementById('valShortTermMax');
const valTruePeakMax = document.getElementById('valTruePeakMax');
const verdictText = document.getElementById('verdictText');

// Graph & Transport DOM
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

// ================= HISTOGRAM HELPER =================
function addHistogramSample(histArray, lufsValue) {
  if (!isFinite(lufsValue) || lufsValue < -60 || lufsValue > 0) return;
  const binIdx = Math.min(HIST_BINS - 1, Math.max(0, Math.floor(((lufsValue + 60) / 60) * HIST_BINS)));
  histArray[binIdx] += 1;
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

// ================= 4. FILE DRAG & DROP + OFFLINE & REALTIME PLAYER =================
async function handleFileLoad(file) {
  if (!file) return;
  stopAllActiveAudio();

  try {
    setStatus('file', 'WAIT');
    fileMetaInfo.textContent = `Loading ${file.name}...`;

    const ctx = ensureAudioContext();
    const arrayBuf = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuf);

    loadedAudioBuffer = audioBuffer;
    loadedFileName = file.name;
    filePauseOffsetSec = 0;

    offlineAnalysisResult = analyzeAudioBufferOffline(audioBuffer);
    offlineHistogram.fill(0);
    for (const pt of offlineAnalysisResult.timeline) {
      addHistogramSample(offlineHistogram, isFinite(pt.shortTerm) ? pt.shortTerm : pt.momentary);
    }

    timeWindowSec = 0;
    windowSelector.value = '0';

    fileMetaInfo.textContent = file.name;
    setStatus('file', 'FILE');
  } catch (err) {
    alert(`오디오 파일을 디코딩할 수 없습니다: ${err.message}`);
    setStatus('standby', '');
  }
}

function startFilePlayback(offsetSec = filePauseOffsetSec) {
  if (!loadedAudioBuffer) return;
  const ctx = ensureAudioContext();

  if (fileSourceNode) {
    try {
      fileSourceNode.onended = null;
      fileSourceNode.stop();
      fileSourceNode.disconnect();
    } catch (_) {}
  }

  fileSourceNode = ctx.createBufferSource();
  fileSourceNode.buffer = loadedAudioBuffer;

  engine.setConfig(loadedAudioBuffer.sampleRate, 2);
  processorNode = ctx.createScriptProcessor(2048, 2, 2);
  zeroGainNode = ctx.createGain();
  zeroGainNode.gain.value = 0;

  processorNode.onaudioprocess = (e) => {
    const inBuf = e.inputBuffer;
    const ch0 = inBuf.getChannelData(0);
    const ch1 = inBuf.numberOfChannels > 1 ? inBuf.getChannelData(1) : ch0;
    engine.process([ch0, ch1]);
  };

  fileSourceNode.connect(ctx.destination);
  fileSourceNode.connect(processorNode);
  processorNode.connect(zeroGainNode);
  zeroGainNode.connect(ctx.destination);

  const clampedOffset = Math.max(0, Math.min(offsetSec, loadedAudioBuffer.duration - 0.05));
  filePauseOffsetSec = clampedOffset;
  filePlayStartCtxTime = ctx.currentTime - clampedOffset;

  fileSourceNode.onended = () => {
    if (fileIsPlaying) {
      fileIsPlaying = false;
      filePauseOffsetSec = 0;
      updatePlayPauseIcon();
    }
  };

  fileSourceNode.start(0, clampedOffset);
  fileIsPlaying = true;
  currentMode = 'file';
  updatePlayPauseIcon();
}

function pauseFilePlayback() {
  if (!fileIsPlaying || !audioCtx) return;
  filePauseOffsetSec = Math.min(
    loadedAudioBuffer.duration,
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
  filePauseOffsetSec = 0;
}

function getFilePlayheadSec() {
  if (!loadedAudioBuffer) return 0;
  if (fileIsPlaying && audioCtx) {
    return Math.min(loadedAudioBuffer.duration, audioCtx.currentTime - filePlayStartCtxTime);
  }
  return filePauseOffsetSec;
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
  if (currentMode === 'file' && offlineAnalysisResult) {
    const summary = offlineAnalysisResult.summary;
    if (fileIsPlaying) {
      const liveSnap = engine.getSnapshot();
      return {
        ...summary,
        momentary: liveSnap.momentary,
        shortTerm: liveSnap.shortTerm,
        truePeakL: liveSnap.truePeakL,
        truePeakR: liveSnap.truePeakR,
        psr: liveSnap.psr,
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

function updateReadouts(snap) {
  const preset = STREAMING_PRESETS[currentPresetKey];

  // 5-Row Slate Stack
  valShortTerm.innerHTML = formatYouleanValue(snap.shortTerm, 'LUFS');
  valIntegrated.innerHTML = formatYouleanValue(snap.integrated, 'LUFS');
  valLRA.innerHTML = `${snap.lra.toFixed(1)} <span class="ylm-unit">LU</span>`;
  valPSR.innerHTML = isFinite(snap.psr) && snap.psr > 0 ? `${snap.psr.toFixed(1)} <span class="ylm-unit">LU</span>` : '-';
  valPLR.innerHTML = isFinite(snap.plr) && snap.plr > 0 ? `${snap.plr.toFixed(1)} <span class="ylm-unit">LU</span>` : '-';

  // 3 Rounded Pills
  valMomentaryMax.innerHTML = formatYouleanValue(snap.momentaryMax, 'LUFS');
  valShortTermMax.innerHTML = formatYouleanValue(snap.shortTermMax, 'LUFS');
  valTruePeakMax.innerHTML = formatYouleanValue(snap.maxTruePeak, 'dB');

  // Highlight True Peak pill if clipping
  if (snap.maxTruePeak > preset.maxTruePeak) {
    valTruePeakMax.style.color = snap.maxTruePeak > 0 ? '#ff6b6b' : '#f6ad55';
  } else {
    valTruePeakMax.style.color = '#ffffff';
  }

  // Update AUTO button label with Target Delta when active
  if (isFinite(snap.integrated) && snap.integrated > -70) {
    const diff = preset.targetLUFS - snap.integrated;
    const sign = diff > 0 ? '+' : '';
    btnAutoScale.textContent = `${sign}${diff.toFixed(1)} LU`;
    verdictText.title = `${preset.name} (${preset.targetLUFS} LUFS): Gain ${sign}${diff.toFixed(1)} dB | True Peak Max: ${formatDB(snap.maxTruePeak, 1)} dBTP`;
  } else {
    btnAutoScale.textContent = 'AUTO';
  }
}

// ================= COLUMN 1: YOULEAN LEFT BAR METER RENDERER =================
function renderBarMeters(snap) {
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

  // Build scale marks including active target (e.g. -23 or -14)
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

    // Left number
    ctx.fillStyle = isTarget ? '#d8a648' : '#e5e9f0';
    ctx.textAlign = 'right';
    ctx.fillText(`${m}`, leftTickX - 4, y);

    // Left tick '-'
    ctx.strokeStyle = isTarget ? '#d8a648' : '#9aa3b2';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(leftTickX, y);
    ctx.lineTo(leftTickX + 7, y);
    ctx.stroke();

    // Right tick '-'
    ctx.beginPath();
    ctx.moveTo(rightTickX - 7, y);
    ctx.lineTo(rightTickX, y);
    ctx.stroke();
  }

  // Draw Momentary Bar Fill (Blue below target, Red above target)
  const momVal = isFinite(snap.momentary) ? snap.momentary : minLUFS;
  if (momVal > minLUFS) {
    const momY = lufsToY(momVal);
    const targetY = lufsToY(preset.targetLUFS);
    const bottomY = topPad + barH;

    // Blue section up to target
    const blueTopY = Math.max(momY, targetY);
    if (bottomY > blueTopY) {
      ctx.fillStyle = '#5887da';
      ctx.fillRect(barX, blueTopY, barW, bottomY - blueTopY);
    }

    // Red section above target
    if (momY < targetY) {
      ctx.fillStyle = '#d95852';
      ctx.fillRect(barX, momY, barW, targetY - momY);
    }
  }

  // Draw LRA Bracket [   ] around Integrated / Short-Term center
  const centerRef = isFinite(snap.integrated) && snap.integrated > -54 ? snap.integrated : -31;
  const halfLRA = Math.max(2, (snap.lra || 4) / 2);
  const brkTopY = lufsToY(centerRef + halfLRA);
  const brkBotY = lufsToY(centerRef - halfLRA);

  ctx.strokeStyle = '#9aa3b2';
  ctx.lineWidth = 1.4;
  // Left '['
  ctx.beginPath();
  ctx.moveTo(barX + 3, brkTopY);
  ctx.lineTo(barX - 1, brkTopY);
  ctx.lineTo(barX - 1, brkBotY);
  ctx.lineTo(barX + 3, brkBotY);
  ctx.stroke();
  // Right ']'
  ctx.beginPath();
  ctx.moveTo(barX + barW - 3, brkTopY);
  ctx.lineTo(barX + barW + 1, brkTopY);
  ctx.lineTo(barX + barW + 1, brkBotY);
  ctx.lineTo(barX + barW - 3, brkBotY);
  ctx.stroke();

  // White Triangle Pointer on the right (◀)
  const ptrVal = isFinite(snap.momentary) && snap.momentary > minLUFS ? snap.momentary : minLUFS + 2;
  const ptrY = lufsToY(ptrVal);
  ctx.fillStyle = '#e5e9f0';
  ctx.beginPath();
  ctx.moveTo(rightTickX - 8, ptrY);
  ctx.lineTo(rightTickX + 4, ptrY - 6);
  ctx.lineTo(rightTickX + 4, ptrY + 6);
  ctx.closePath();
  ctx.fill();

  // Bottom 'M' and 'LUFS' labels under the bar
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
  ctx.fillText('M', barCenterX, footerLineY + 9);

  ctx.font = '600 14px "Open Sans", sans-serif';
  ctx.fillStyle = '#ffffff';
  ctx.fillText('LUFS', barCenterX, footerLineY + 25);

  ctx.restore();
}

// ================= COLUMN 3 & 4: HISTOGRAM + TIME GRAPH + MINIMAP RENDERER =================
function renderHistoryGraph(snap) {
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

  // Vertical layout: [Main Plot] [Time Axis Row (26px)] [Bottom Minimap Strip (34px)]
  const topPad = 14;
  const minimapH = 32;
  const timeRowH = 28;
  const botPad = minimapH + timeRowH + 8;
  const plotH = Math.max(30, H - topPad - botPad);

  // Horizontal layout: [Histogram Zone (108px)] [Scale Gutter (44px)] [Time-Series Plot]
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

  // 1. Horizontal Grid Lines & Scale Labels (for both Histogram & Main Graph)
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
      // Histogram grid line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(6, y);
      ctx.lineTo(histAxisX, y);
      ctx.stroke();

      // Main graph grid line
      ctx.strokeStyle = isTarget ? 'rgba(216, 166, 72, 0.22)' : 'rgba(255, 255, 255, 0.055)';
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotRight, y);
      ctx.stroke();
    }

    // Scale number & tick at left of main graph
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

  // 2. Render Loudness Distribution Histogram (Left of histAxisX)
  const activeHist = currentMode === 'file' && offlineAnalysisResult ? offlineHistogram : liveHistogram;
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

  // Vertical Histogram Axis Line: Red above targetY, Blue below targetY!
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
  const isFileMode = currentMode === 'file' && offlineAnalysisResult;
  const sourceTimeline = isFileMode ? offlineAnalysisResult.timeline : liveHistory;

  const lastTime = sourceTimeline.length > 0 ? sourceTimeline[sourceTimeline.length - 1].timeSec : 0;
  const totalDuration = isFileMode
    ? offlineAnalysisResult.durationSec
    : Math.max(timeWindowSec || 30, lastTime);

  let startTime = 0;
  let endTime = totalDuration;

  if (timeWindowSec > 0) {
    if (isFileMode && fileIsPlaying) {
      const playhead = getFilePlayheadSec();
      endTime = Math.max(timeWindowSec, playhead);
      startTime = endTime - timeWindowSec;
    } else if (!isFileMode) {
      endTime = Math.max(timeWindowSec, lastTime);
      startTime = endTime - timeWindowSec;
    } else {
      endTime = Math.min(totalDuration, timeWindowSec);
      startTime = 0;
    }
  }

  const spanSec = Math.max(1, endTime - startTime);
  const timeToX = (t) => plotLeft + ((t - startTime) / spanSec) * plotW;

  if (sourceTimeline.length > 1) {
    const visible = sourceTimeline.filter((pt) => pt.timeSec >= startTime && pt.timeSec <= endTime);

    if (visible.length > 1) {
      const getVal = (pt) => (layers.M && !layers.S ? pt.momentary : pt.shortTerm);

      // Build main loudness polygon path
      const drawCurvePath = () => {
        ctx.beginPath();
        ctx.moveTo(timeToX(visible[0].timeSec), topPad + plotH);
        for (let i = 0; i < visible.length; i++) {
          ctx.lineTo(timeToX(visible[i].timeSec), lufsToY(getVal(visible[i])));
        }
        ctx.lineTo(timeToX(visible[visible.length - 1].timeSec), topPad + plotH);
        ctx.closePath();
      };

      // Below Target Clip -> Blue Fill & Stroke
      ctx.save();
      ctx.beginPath();
      ctx.rect(plotLeft, targetY, plotW, topPad + plotH - targetY);
      ctx.clip();
      drawCurvePath();
      ctx.fillStyle = 'rgba(88, 135, 218, 0.45)';
      ctx.fill();
      ctx.strokeStyle = '#6b9be8';
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.restore();

      // Above Target Clip -> Red Fill & Stroke (Youlean over-target highlight!)
      ctx.save();
      ctx.beginPath();
      ctx.rect(plotLeft, topPad, plotW, Math.max(0, targetY - topPad));
      ctx.clip();
      drawCurvePath();
      ctx.fillStyle = 'rgba(217, 88, 82, 0.55)';
      ctx.fill();
      ctx.strokeStyle = '#ef6b65';
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.restore();

      // Optional Secondary Momentary Line if both S and M are active
      if (layers.S && layers.M) {
        ctx.beginPath();
        for (let i = 0; i < visible.length; i++) {
          const x = timeToX(visible[i].timeSec);
          const y = lufsToY(visible[i].momentary);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
        ctx.lineWidth = 1;
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
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(px, topPad);
          ctx.lineTo(px, topPad + plotH);
          ctx.stroke();
        }
      }
    }
  }

  // 4. Time Axis Labels (e.g. 18s, 20s, 22s, 24s, 26s, 28s)
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

  // 5. Bottom Minimap Timeline Strip (Exact Youlean Bottom Overview Bar!)
  const miniTop = H - minimapH - 4;
  const miniLeft = plotLeft - 36;
  const miniW = plotRight - miniLeft;

  // Minimap left bracket '['
  ctx.strokeStyle = '#5c6270';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(miniLeft + 4, miniTop + 2);
  ctx.lineTo(miniLeft, miniTop + 2);
  ctx.lineTo(miniLeft, miniTop + minimapH - 2);
  ctx.lineTo(miniLeft + 4, miniTop + minimapH - 2);
  ctx.stroke();

  // Minimap full-session loudness blocks in slate-blue (#4b6898)
  if (sourceTimeline.length > 1) {
    const fullDur = Math.max(1, sourceTimeline[sourceTimeline.length - 1].timeSec);
    ctx.beginPath();
    ctx.moveTo(miniLeft + 8, miniTop + minimapH - 2);
    const step = Math.max(1, Math.floor(sourceTimeline.length / 200));
    for (let i = 0; i < sourceTimeline.length; i += step) {
      const pt = sourceTimeline[i];
      const mx = miniLeft + 8 + (pt.timeSec / fullDur) * (miniW - 16);
      const val = isFinite(pt.shortTerm) ? pt.shortTerm : pt.momentary;
      const norm = Math.max(0, Math.min(1, (val - minLUFS) / (maxLUFS - minLUFS)));
      const my = miniTop + minimapH - 2 - norm * (minimapH - 6);
      ctx.lineTo(mx, my);
    }
    const lastPt = sourceTimeline[sourceTimeline.length - 1];
    ctx.lineTo(miniLeft + 8 + (lastPt.timeSec / fullDur) * (miniW - 16), miniTop + minimapH - 2);
    ctx.closePath();
    ctx.fillStyle = 'rgba(88, 135, 218, 0.55)';
    ctx.fill();
    ctx.strokeStyle = '#6b9be8';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Minimap Viewport Box on the right (like the gray bordered box in the screenshot)
  const vpBoxW = Math.min(185, miniW * 0.34);
  const vpBoxX = plotRight - vpBoxW;
  ctx.fillStyle = 'rgba(70, 75, 87, 0.65)';
  ctx.strokeStyle = '#7c828f';
  ctx.lineWidth = 1.5;
  ctx.fillRect(vpBoxX, miniTop, vpBoxW, minimapH - 2);
  ctx.strokeRect(vpBoxX, miniTop, vpBoxW, minimapH - 2);

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
    const labelText = `${formatTime(closest.timeSec)} | S: ${formatLUFS(closest.shortTerm, 1)} LUFS | M: ${formatLUFS(closest.momentary, 1)} LUFS | TP: ${formatDB(tpPeak, 1)} dB`;
    ctx.font = '700 11.5px "Open Sans", sans-serif';
    const textW = ctx.measureText(labelText).width + 18;
    const bx = Math.min(plotRight - textW - 4, Math.max(plotLeft + 4, cx - textW / 2));
    const by = topPad + 6;

    ctx.fillStyle = 'rgba(28, 31, 38, 0.92)';
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

// ================= EXPORT REPORT =================
function exportLoudnessReport() {
  const snap = getActiveSnapshot();
  const preset = STREAMING_PRESETS[currentPresetKey];
  const report = {
    generatedAt: new Date().toISOString(),
    standard: 'ITU-R BS.1770-4 / EBU R128',
    sourceMode: currentMode,
    fileName: loadedFileName || null,
    targetPreset: preset,
    metrics: {
      integratedLUFS: isFinite(snap.integrated) ? Number(snap.integrated.toFixed(2)) : null,
      shortTermMaxLUFS: isFinite(snap.shortTermMax) ? Number(snap.shortTermMax.toFixed(2)) : null,
      momentaryMaxLUFS: isFinite(snap.momentaryMax) ? Number(snap.momentaryMax.toFixed(2)) : null,
      loudnessRangeLU: Number(snap.lra.toFixed(2)),
      truePeakMaxDBTP: isFinite(snap.maxTruePeak) ? Number(snap.maxTruePeak.toFixed(2)) : null,
      plrLU: isFinite(snap.plr) ? Number(snap.plr.toFixed(2)) : null,
      psrLU: isFinite(snap.psr) ? Number(snap.psr.toFixed(2)) : null,
      truePeakClips: snap.clipCount,
      durationSec: Number(snap.durationSec.toFixed(2)),
    },
  };

  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `loudness-report-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
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
  liveHistory = [];
  liveHistogram.fill(0);
  if (currentMode === 'file' && !fileIsPlaying) {
    offlineAnalysisResult = null;
    offlineHistogram.fill(0);
    loadedAudioBuffer = null;
    loadedFileName = '';
    setStatus('standby', '');
    fileMetaInfo.textContent = 'READY';
  }
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

fileInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if (file) handleFileLoad(file);
  fileInput.value = '';
});

presetSelect.addEventListener('change', (e) => {
  currentPresetKey = e.target.value;
});

btnReset.addEventListener('click', resetMeters);

btnMiniMode.addEventListener('click', () => {
  document.body.classList.toggle('mini-mode');
  btnMiniMode.classList.toggle('active', document.body.classList.contains('mini-mode'));
});

btnPiP.addEventListener('click', toggleAlwaysOnTopPiP);
btnExport.addEventListener('click', exportLoudnessReport);

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
    btn.classList.toggle('active', layers[key]);
  });
});

// Floating graph buttons
if (btnFloatAudio) {
  btnFloatAudio.addEventListener('click', toggleReferenceTone);
}
if (btnFloatPeaks) {
  btnFloatPeaks.addEventListener('click', () => {
    layers.T = !layers.T;
    btnFloatPeaks.classList.toggle('active-red', layers.T);
  });
}
if (btnFloatTarget) {
  btnFloatTarget.addEventListener('click', () => {
    layers.G = !layers.G;
    btnFloatTarget.classList.toggle('active-red', layers.G);
  });
}

btnPlayPause.addEventListener('click', () => {
  if (loadedAudioBuffer) {
    if (fileIsPlaying) pauseFilePlayback();
    else startFilePlayback(filePauseOffsetSec);
  } else {
    startLiveDeviceInput();
  }
});

btnStopFile.addEventListener('click', () => {
  if (loadedAudioBuffer) stopFilePlayback();
  else stopAllActiveAudio();
});

historyCanvas.addEventListener('mousemove', (e) => {
  const rect = historyCanvas.getBoundingClientRect();
  hoverPoint = { x: e.clientX - rect.left, y: e.clientY - rect.top };
});

historyCanvas.addEventListener('mouseleave', () => {
  hoverPoint = null;
});

historyCanvas.addEventListener('click', (e) => {
  if (currentMode !== 'file' || !loadedAudioBuffer || !offlineAnalysisResult) return;
  const rect = historyCanvas.getBoundingClientRect();
  const plotLeft = 152;
  const plotRight = rect.width - 12;
  const plotW = Math.max(20, plotRight - plotLeft);
  const clickX = e.clientX - rect.left;
  if (clickX < plotLeft || clickX > plotRight) return;

  const ratio = (clickX - plotLeft) / plotW;
  const targetSec = ratio * loadedAudioBuffer.duration;
  filePauseOffsetSec = targetSec;
  if (fileIsPlaying) {
    startFilePlayback(targetSec);
  }
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
  if (e.code === 'Space' && loadedAudioBuffer) {
    e.preventDefault();
    btnPlayPause.click();
  } else if (e.key === 'r' || e.key === 'R') {
    resetMeters();
  } else if (e.key === 'm' || e.key === 'M') {
    btnMiniMode.click();
  }
});

// ================= MAIN 60FPS RENDER LOOP =================
function tick() {
  const snap = getActiveSnapshot();
  updateReadouts(snap);
  renderHistoryGraph(snap);
  renderBarMeters(snap);
  requestAnimationFrame(tick);
}

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
