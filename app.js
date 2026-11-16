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
let timeWindowSec = 60; // 30, 60, 300, or 0 (full)
let graphPaused = false;

// Live history buffer (10 Hz points)
let liveHistory = [];
let liveStartTime = 0;

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
let fileSourceNode = null;
let fileIsPlaying = false;
let filePlayStartCtxTime = 0;
let filePauseOffsetSec = 0;

// Graph interaction state
let hoverPoint = null; // { x, y } in CSS pixels

// ================= DOM ELEMENTS =================
const elStatusBadge = document.getElementById('statusBadge');
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

// Readout DOM
const valIntegrated = document.getElementById('valIntegrated');
const lblTargetLUFS = document.getElementById('lblTargetLUFS');
const valTargetDelta = document.getElementById('valTargetDelta');
const valShortTerm = document.getElementById('valShortTerm');
const valShortTermMax = document.getElementById('valShortTermMax');
const valMomentary = document.getElementById('valMomentary');
const valMomentaryMax = document.getElementById('valMomentaryMax');
const valLRA = document.getElementById('valLRA');
const valDynamicsTag = document.getElementById('valDynamicsTag');
const valPLR = document.getElementById('valPLR');
const valSamplePeak = document.getElementById('valSamplePeak');
const valTruePeakMax = document.getElementById('valTruePeakMax');
const clipBadge = document.getElementById('clipBadge');
const valTPL = document.getElementById('valTPL');
const valTPR = document.getElementById('valTPR');
const lblCeilingTP = document.getElementById('lblCeilingTP');
const valClipCount = document.getElementById('valClipCount');
const verdictPresetName = document.getElementById('verdictPresetName');
const verdictText = document.getElementById('verdictText');

// Graph & Transport DOM
const graphTargetLabel = document.getElementById('graphTargetLabel');
const windowSelector = document.getElementById('windowSelector');
const btnPauseGraph = document.getElementById('btnPauseGraph');
const historyCanvas = document.getElementById('historyCanvas');
const barMeterCanvas = document.getElementById('barMeterCanvas');
const dropZone = document.getElementById('dropZone');
const dropOverlay = document.getElementById('dropOverlay');

const btnPlayPause = document.getElementById('btnPlayPause');
const iconPlay = document.getElementById('iconPlay');
const btnStopFile = document.getElementById('btnStopFile');
const curTimeEl = document.getElementById('curTime');
const totTimeEl = document.getElementById('totTime');
const seekSlider = document.getElementById('seekSlider');
const fileMetaInfo = document.getElementById('fileMetaInfo');

const barFooterLUFS = document.getElementById('barFooterLUFS');
const barFooterTP = document.getElementById('barFooterTP');

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
    // Record 10 Hz points for realtime graph
    if (snap.durationSec - lastHistoryPushSec >= 0.095) {
      lastHistoryPushSec = snap.durationSec;
      if (!graphPaused) {
        liveHistory.push({
          timeSec: snap.durationSec,
          momentary: snap.momentary,
          shortTerm: snap.shortTerm,
          integrated: snap.integrated,
          truePeakL: snap.truePeakL,
          truePeakR: snap.truePeakR,
        });
        // Keep up to 15 minutes (9000 points at 10Hz) in live ring
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

    if (audioInputs.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = '기본 오디오 입력 장치';
      deviceSelect.appendChild(opt);
      return;
    }

    audioInputs.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      opt.textContent = dev.label || `오디오 입력 장치 #${idx + 1}`;
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
    setStatus('standby', 'STANDBY');
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

    sourceNode = ctx.createMediaStreamSource(activeStream);
    attachRealtimeProcessor(sourceNode, false);

    btnLiveInput.classList.add('active');
    const trackLabel = activeStream.getAudioTracks()[0]?.label || 'Live Audio Input';
    fileMetaInfo.textContent = `실시간 입력: ${trackLabel} (${ctx.sampleRate} Hz)`;
    setStatus('live', 'LIVE INPUT');
  } catch (err) {
    alert(`오디오 입력 장치에 접근할 수 없습니다: ${err.message}`);
  }
}

// ================= 2. SYSTEM / TAB AUDIO CAPTURE =================
async function startSystemAudioCapture() {
  if (currentMode === 'system') {
    stopAllActiveAudio();
    setStatus('standby', 'STANDBY');
    return;
  }

  if (!navigator.mediaDevices?.getDisplayMedia) {
    alert('현재 브라우저는 시스템/화면 오디오 캡처(getDisplayMedia)를 지원하지 않습니다. Chrome 또는 Edge를 사용해 주세요.');
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
      alert(
        '오디오 트랙이 감지되지 않았습니다.\n화면/탭 공유 창 하단의 "시스템 오디오 공유(Share system audio)" 또는 "탭 오디오 공유" 체크박스를 켜주세요.'
      );
      return;
    }

    audioTracks[0].onended = () => {
      stopAllActiveAudio();
      setStatus('standby', 'STANDBY');
    };

    engine.setConfig(ctx.sampleRate, 2);
    liveHistory = [];

    sourceNode = ctx.createMediaStreamSource(activeStream);
    attachRealtimeProcessor(sourceNode, false);

    btnSystemAudio.classList.add('active');
    fileMetaInfo.textContent = `시스템/탭 오디오 캡처 중 (${ctx.sampleRate} Hz)`;
    setStatus('system', 'SYSTEM AUDIO');
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
    setStatus('standby', 'STANDBY');
    return;
  }

  stopAllActiveAudio();
  const ctx = ensureAudioContext();
  engine.setConfig(ctx.sampleRate, 2);
  liveHistory = [];

  // Calculate exact K-weighting gain at 1000 Hz so stereo sine wave measures exactly -14.0 LUFS
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
  // Target LUFS = -14.0 => -0.691 + 10 * log10(2 * (A^2 / 2) * H2) = -14.0
  // => A^2 * H2 = 10^((-14.0 + 0.691) / 10)
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
  fileMetaInfo.textContent = `보정 테스트 신호: 1 kHz Sine (-14.0 LUFS 기준 톤)`;
  setStatus('tone', 'TEST TONE');
}

// ================= 4. FILE DRAG & DROP + OFFLINE & REALTIME PLAYER =================
async function handleFileLoad(file) {
  if (!file) return;
  stopAllActiveAudio();

  try {
    setStatus('file', 'ANALYZING...');
    fileMetaInfo.textContent = `파일 분석 중: ${file.name}...`;

    const ctx = ensureAudioContext();
    const arrayBuf = await file.arrayBuffer();
    const audioBuffer = await ctx.decodeAudioData(arrayBuf);

    loadedAudioBuffer = audioBuffer;
    loadedFileName = file.name;
    filePauseOffsetSec = 0;

    // Perform instant full-file ITU-R BS.1770-4 offline analysis
    const t0 = performance.now();
    offlineAnalysisResult = analyzeAudioBufferOffline(audioBuffer);
    const elapsedMs = Math.round(performance.now() - t0);

    // Switch time window to Full File automatically so user sees complete track contour
    timeWindowSec = 0;
    document.querySelectorAll('#windowSelector .seg-btn').forEach((b) => {
      b.classList.toggle('active', Number(b.dataset.sec) === 0);
    });

    // Enable transport controls
    btnPlayPause.disabled = false;
    btnStopFile.disabled = false;
    seekSlider.disabled = false;
    seekSlider.value = 0;
    curTimeEl.textContent = formatTime(0);
    totTimeEl.textContent = formatTime(audioBuffer.duration);

    fileMetaInfo.textContent = `${file.name} (${(audioBuffer.sampleRate / 1000).toFixed(1)}kHz, ${audioBuffer.numberOfChannels}ch, ${ elapsedMs }ms 만에 전체 분석 완료)`;
    setStatus('file', 'FILE ANALYZED');
  } catch (err) {
    alert(`오디오 파일을 디코딩할 수 없습니다: ${err.message}`);
    setStatus('standby', 'STANDBY');
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

  // Also feed live engine during playback so vertical bar meters dance smoothly at 60fps
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
      seekSlider.value = 0;
      curTimeEl.textContent = formatTime(0);
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
  seekSlider.value = 0;
  curTimeEl.textContent = formatTime(0);
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
    iconPlay.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  } else {
    iconPlay.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  }
}

// ================= STATUS & UI UPDATES =================
function setStatus(mode, label) {
  currentMode = mode;
  elStatusText.textContent = label;
  elStatusBadge.className = 'status-badge';
  if (mode === 'live' || mode === 'system' || mode === 'tone') {
    elStatusBadge.classList.add('live');
  } else if (mode === 'file') {
    elStatusBadge.classList.add('file');
  } else {
    elStatusBadge.classList.add('standby');
  }
}

function getActiveSnapshot() {
  if (currentMode === 'file' && offlineAnalysisResult) {
    const summary = offlineAnalysisResult.summary;
    if (fileIsPlaying) {
      // Combine full-file Integrated/LRA/Max summary with live instantaneous Momentary/ShortTerm/Bar peaks
      const liveSnap = engine.getSnapshot();
      return {
        ...summary,
        momentary: liveSnap.momentary,
        shortTerm: liveSnap.shortTerm,
        truePeakL: liveSnap.truePeakL,
        truePeakR: liveSnap.truePeakR,
      };
    }
    return summary;
  }
  return engine.getSnapshot();
}

function updateReadouts(snap) {
  const preset = STREAMING_PRESETS[currentPresetKey];

  lblTargetLUFS.textContent = `${preset.targetLUFS.toFixed(1)} LUFS`;
  lblCeilingTP.textContent = `${preset.maxTruePeak.toFixed(1)} dBTP`;
  graphTargetLabel.textContent = preset.targetLUFS.toFixed(0);
  verdictPresetName.textContent = preset.name;

  // Integrated LUFS + Target Delta
  valIntegrated.textContent = formatLUFS(snap.integrated, 1);
  if (isFinite(snap.integrated) && snap.integrated > -90) {
    const delta = snap.integrated - preset.targetLUFS;
    const sign = delta > 0 ? '+' : '';
    valTargetDelta.textContent = `Δ ${sign}${delta.toFixed(1)} LU`;
    valTargetDelta.className = 'delta-pill';
    if (Math.abs(delta) <= preset.tolerance) {
      valTargetDelta.classList.add('good');
    } else if (delta > preset.tolerance) {
      valTargetDelta.classList.add('hot');
    } else {
      valTargetDelta.classList.add('warn');
    }
  } else {
    valTargetDelta.textContent = 'Δ -- LU';
    valTargetDelta.className = 'delta-pill';
  }

  // Short-Term & Momentary
  valShortTerm.textContent = formatLUFS(snap.shortTerm, 1);
  valShortTermMax.textContent = `${formatLUFS(snap.shortTermMax, 1)} LUFS`;
  valMomentary.textContent = formatLUFS(snap.momentary, 1);
  valMomentaryMax.textContent = `${formatLUFS(snap.momentaryMax, 1)} LUFS`;

  // LRA & Dynamics Tag
  valLRA.textContent = snap.lra.toFixed(1);
  if (!isFinite(snap.integrated) || snap.integrated < -70) {
    valDynamicsTag.textContent = '--';
  } else if (snap.lra < 4) {
    valDynamicsTag.textContent = '압축됨 (Tight)';
  } else if (snap.lra <= 10) {
    valDynamicsTag.textContent = '균형적 (Balanced)';
  } else {
    valDynamicsTag.textContent = '넓은 다이내믹 (Wide)';
  }

  // PLR & Sample Peak
  valPLR.textContent = isFinite(snap.plr) && snap.plr > 0 ? snap.plr.toFixed(1) : '--';
  valSamplePeak.textContent = `${formatDB(snap.maxSamplePeak, 1)} dBFS`;

  // True Peak Max & Clip Indicator
  valTruePeakMax.textContent = formatDB(snap.maxTruePeak, 1);
  valTPL.textContent = formatDB(snap.truePeakL, 1);
  valTPR.textContent = formatDB(snap.truePeakR, 1);
  valClipCount.textContent = `${snap.clipCount} clips (>0 dBTP)`;

  clipBadge.className = 'clip-indicator';
  if (snap.maxTruePeak > 0.0) {
    clipBadge.textContent = 'CLIP (>0 dBTP)';
    clipBadge.classList.add('clip');
  } else if (snap.maxTruePeak > preset.maxTruePeak) {
    clipBadge.textContent = `OVER ${preset.maxTruePeak.toFixed(1)}`;
    clipBadge.classList.add('warn');
  } else {
    clipBadge.textContent = 'SAFE';
  }

  // Streaming Verdict Text
  if (isFinite(snap.integrated) && snap.integrated > -70) {
    const diff = preset.targetLUFS - snap.integrated;
    const tpWarning =
      snap.maxTruePeak > preset.maxTruePeak
        ? ` ⚠️ 트루 피크가 권장치(${preset.maxTruePeak.toFixed(1)} dBTP)를 ${(snap.maxTruePeak - preset.maxTruePeak).toFixed(1)} dB 초과하여 손실 인코딩 시 클리핑이 발생할 수 있습니다.`
        : ` ✅ 트루 피크 여유 공간이 안전합니다 (${snap.maxTruePeak.toFixed(1)} dBTP).`;

    if (diff < -0.2) {
      verdictText.innerHTML = `<strong>${preset.name}</strong>에서 재생 시 볼륨이 약 <strong>${diff.toFixed(1)} dB</strong> 감쇠(Turned Down)됩니다.${tpWarning}`;
    } else if (diff > 0.5) {
      verdictText.innerHTML = `타겟보다 <strong>${diff.toFixed(1)} LU</strong> 낮습니다. 리미터 여유 범위 내에서 음량을 높일 수 있습니다.${tpWarning}`;
    } else {
      verdictText.innerHTML = `🎯 <strong>${preset.name}</strong> 타겟(${preset.targetLUFS} LUFS)에 정확히 부합합니다!${tpWarning}`;
    }
  }

  // Update file playback scrubber if playing
  if (currentMode === 'file' && loadedAudioBuffer) {
    const playSec = getFilePlayheadSec();
    curTimeEl.textContent = formatTime(playSec);
    if (fileIsPlaying) {
      seekSlider.value = Math.round((playSec / loadedAudioBuffer.duration) * 1000);
    }
  }

  // Bar meter footer labels
  barFooterLUFS.textContent = formatLUFS(snap.momentary, 1);
  barFooterTP.textContent = formatDB(Math.max(snap.truePeakL, snap.truePeakR), 1);
}

// ================= HIGH-DPI CANVAS GRAPH RENDERER =================
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

  // Clear background
  ctx.fillStyle = '#0b0e14';
  ctx.fillRect(0, 0, W, H);

  const padLeft = 42;
  const padRight = 14;
  const padTop = 22;
  const padBottom = 24;
  const plotW = Math.max(10, W - padLeft - padRight);
  const plotH = Math.max(10, H - padTop - padBottom);

  // LUFS Y-Axis scale: -60 LUFS (bottom) to 0 LUFS (top)
  const minLUFS = -54;
  const maxLUFS = 0;
  const lufsToY = (lufs) => {
    const clamped = Math.max(minLUFS, Math.min(maxLUFS, isFinite(lufs) ? lufs : minLUFS));
    const norm = (clamped - minLUFS) / (maxLUFS - minLUFS);
    return padTop + plotH * (1 - norm);
  };

  // Horizontal Grid Lines
  const gridSteps = [0, -6, -9, -14, -16, -18, -23, -27, -36, -48];
  ctx.font = '10px "SF Mono", Menlo, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  for (const db of gridSteps) {
    const y = lufsToY(db);
    ctx.strokeStyle = db === 0 ? 'rgba(239, 68, 68, 0.3)' : 'rgba(255, 255, 255, 0.055)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + plotW, y);
    ctx.stroke();

    ctx.fillStyle = '#637286';
    ctx.fillText(`${db}`, padLeft - 6, y);
  }

  // Target Preset Line & Tolerance Band
  const preset = STREAMING_PRESETS[currentPresetKey];
  const targetY = lufsToY(preset.targetLUFS);
  const bandTopY = lufsToY(preset.targetLUFS + preset.tolerance);
  const bandBotY = lufsToY(preset.targetLUFS - preset.tolerance);

  ctx.fillStyle = 'rgba(16, 185, 129, 0.07)';
  ctx.fillRect(padLeft, bandTopY, plotW, Math.max(2, bandBotY - bandTopY));

  ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
  ctx.setLineDash([5, 4]);
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padLeft, targetY);
  ctx.lineTo(padLeft + plotW, targetY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Select data points to plot
  const isFileMode = currentMode === 'file' && offlineAnalysisResult;
  const sourceTimeline = isFileMode ? offlineAnalysisResult.timeline : liveHistory;

  if (sourceTimeline.length > 1) {
    const lastTime = sourceTimeline[sourceTimeline.length - 1].timeSec;
    const totalDuration = isFileMode ? offlineAnalysisResult.durationSec : Math.max(timeWindowSec || lastTime, lastTime);

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
    const timeToX = (t) => padLeft + ((t - startTime) / spanSec) * plotW;

    // Filter visible slice
    const visible = sourceTimeline.filter((pt) => pt.timeSec >= startTime && pt.timeSec <= endTime);

    if (visible.length > 1) {
      // 1. Momentary Shaded Area (400ms)
      ctx.beginPath();
      ctx.moveTo(timeToX(visible[0].timeSec), padTop + plotH);
      for (let i = 0; i < visible.length; i++) {
        const pt = visible[i];
        ctx.lineTo(timeToX(pt.timeSec), lufsToY(pt.momentary));
      }
      ctx.lineTo(timeToX(visible[visible.length - 1].timeSec), padTop + plotH);
      ctx.closePath();

      const grad = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
      grad.addColorStop(0, 'rgba(59, 130, 246, 0.36)');
      grad.addColorStop(0.65, 'rgba(0, 212, 255, 0.14)');
      grad.addColorStop(1, 'rgba(0, 212, 255, 0.02)');
      ctx.fillStyle = grad;
      ctx.fill();

      // 2. Short-Term LUFS Line (3s)
      ctx.beginPath();
      for (let i = 0; i < visible.length; i++) {
        const pt = visible[i];
        const x = timeToX(pt.timeSec);
        const y = lufsToY(pt.shortTerm);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 1.8;
      ctx.stroke();

      // 3. Integrated LUFS Line
      if (isFinite(snap.integrated) && snap.integrated > minLUFS) {
        const intY = lufsToY(snap.integrated);
        ctx.strokeStyle = 'rgba(168, 85, 247, 0.85)';
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.moveTo(padLeft, intY);
        ctx.lineTo(padLeft + plotW, intY);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // 4. True Peak Over-Threshold Markers along Top Strip
      for (let i = 0; i < visible.length; i++) {
        const pt = visible[i];
        const maxTP = Math.max(pt.truePeakL, pt.truePeakR);
        if (maxTP > preset.maxTruePeak) {
          const x = timeToX(pt.timeSec);
          ctx.fillStyle = maxTP > 0 ? '#ef4444' : '#f59e0b';
          ctx.fillRect(x - 1, padTop - 10, 2.5, 7);
        }
      }

      // 5. File Playhead Line
      if (isFileMode) {
        const playSec = getFilePlayheadSec();
        if (playSec >= startTime && playSec <= endTime) {
          const px = timeToX(playSec);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(px, padTop - 6);
          ctx.lineTo(px, padTop + plotH);
          ctx.stroke();
        }
      }

      // 6. Time Axis Labels
      ctx.fillStyle = '#637286';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const numTimeTicks = 6;
      for (let t = 0; t <= numTimeTicks; t++) {
        const frac = t / numTimeTicks;
        const sec = startTime + frac * spanSec;
        const tx = padLeft + frac * plotW;
        ctx.fillText(formatTime(sec), tx, padTop + plotH + 6);
      }

      // 7. Interactive Hover Crosshair & Readout Tooltip
      if (
        hoverPoint &&
        hoverPoint.x >= padLeft &&
        hoverPoint.x <= padLeft + plotW &&
        hoverPoint.y >= padTop &&
        hoverPoint.y <= padTop + plotH
      ) {
        const hoverRatio = (hoverPoint.x - padLeft) / plotW;
        const hoverSec = startTime + hoverRatio * spanSec;

        // Find closest timeline point
        let closest = visible[0];
        let minDiff = Infinity;
        for (let i = 0; i < visible.length; i++) {
          const d = Math.abs(visible[i].timeSec - hoverSec);
          if (d < minDiff) {
            minDiff = d;
            closest = visible[i];
          }
        }

        const cx = timeToX(closest.timeSec);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(cx, padTop);
        ctx.lineTo(cx, padTop + plotH);
        ctx.stroke();

        // Draw Tooltip Box
        const tpPeak = Math.max(closest.truePeakL, closest.truePeakR);
        const lines = [
          `Time: ${formatTime(closest.timeSec)}`,
          `Short-Term: ${formatLUFS(closest.shortTerm, 1)} LUFS`,
          `Momentary:  ${formatLUFS(closest.momentary, 1)} LUFS`,
          `True Peak:  ${formatDB(tpPeak, 1)} dBTP`,
        ];

        const boxW = 162;
        const boxH = 68;
        const bx = cx + 12 + boxW > padLeft + plotW ? cx - boxW - 12 : cx + 12;
        const by = Math.max(padTop + 4, Math.min(hoverPoint.y - boxH / 2, padTop + plotH - boxH - 4));

        ctx.fillStyle = 'rgba(15, 20, 28, 0.92)';
        ctx.strokeStyle = '#35445e';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(bx, by, boxW, boxH, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#e2e8f0';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.font = '10.5px "SF Mono", Menlo, monospace';
        lines.forEach((line, idx) => {
          ctx.fillText(line, bx + 8, by + 7 + idx * 14);
        });
      }
    }
  } else {
    // Empty state watermark inside graph
    ctx.fillStyle = 'rgba(154, 167, 184, 0.35)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '13px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillText(
      '오디오 파일을 여기에 드래그 앤 드롭하거나 상단에서 입력 소스를 시작하세요',
      padLeft + plotW / 2,
      padTop + plotH / 2
    );
  }

  ctx.restore();
}

// ================= VERTICAL BAR METERS RENDERER =================
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

  const topPad = 8;
  const botPad = 6;
  const barH = Math.max(10, H - topPad - botPad);

  const minDB = -60;
  const maxDB = 3;
  const dbToY = (val) => {
    const clamped = Math.max(minDB, Math.min(maxDB, isFinite(val) ? val : minDB));
    const norm = (clamped - minDB) / (maxDB - minDB);
    return topPad + barH * (1 - norm);
  };

  // Layout columns: [LUFS Bar] [Scale Labels] [TP L] [TP R]
  const lufsX = 6;
  const lufsW = 24;
  const tpLX = W - 36;
  const tpRX = W - 18;
  const tpW = 13;

  // Draw backgrounds
  ctx.fillStyle = '#0b0e14';
  ctx.fillRect(lufsX, topPad, lufsW, barH);
  ctx.fillRect(tpLX, topPad, tpW, barH);
  ctx.fillRect(tpRX, topPad, tpW, barH);

  // 1. LUFS Momentary Bar
  const preset = STREAMING_PRESETS[currentPresetKey];
  const momY = dbToY(snap.momentary);
  const fillH = Math.max(0, topPad + barH - momY);

  const lufsGrad = ctx.createLinearGradient(0, topPad + barH, 0, topPad);
  lufsGrad.addColorStop(0, '#0284c7');
  lufsGrad.addColorStop(0.65, '#00d4ff');
  lufsGrad.addColorStop(0.82, '#10b981');
  lufsGrad.addColorStop(0.92, '#f59e0b');
  lufsGrad.addColorStop(1, '#ef4444');

  ctx.fillStyle = lufsGrad;
  ctx.fillRect(lufsX, momY, lufsW, fillH);

  // Short-Term tick marker on LUFS bar
  if (isFinite(snap.shortTerm) && snap.shortTerm > minDB) {
    const stY = dbToY(snap.shortTerm);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(lufsX - 2, stY - 1, lufsW + 4, 2.5);
  }

  // Target LUFS tick on LUFS bar
  const targetY = dbToY(preset.targetLUFS);
  ctx.fillStyle = '#10b981';
  ctx.fillRect(lufsX - 3, targetY - 1, lufsW + 6, 2);

  // 2. True Peak L & R Bars
  const drawTPBar = (x, dbVal, maxChVal) => {
    const y = dbToY(dbVal);
    const h = Math.max(0, topPad + barH - y);
    ctx.fillStyle = dbVal > 0 ? '#ef4444' : dbVal > preset.maxTruePeak ? '#f59e0b' : '#3b82f6';
    ctx.fillRect(x, y, tpW, h);

    // Top Clip LED
    ctx.fillStyle = maxChVal > 0 ? '#ef4444' : 'rgba(255,255,255,0.08)';
    ctx.fillRect(x, 0, tpW, 5);
  };

  drawTPBar(tpLX, snap.truePeakL, snap.maxTruePeak);
  drawTPBar(tpRX, snap.truePeakR, snap.maxTruePeak);

  // 3. Scale Markings in center
  const scaleMarks = [0, -6, -9, -14, -18, -23, -30, -40, -50];
  ctx.fillStyle = '#637286';
  ctx.font = '9px "SF Mono", Menlo, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const centerScaleX = (lufsX + lufsW + tpLX) / 2;

  for (const m of scaleMarks) {
    const y = dbToY(m);
    ctx.fillText(`${m}`, centerScaleX, y);
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
        width: 430,
        height: 540,
      });

      // Copy stylesheets into PiP window
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

  // Fallback: Toggle Mini Mode and alert instructions
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
      samplePeakMaxDBFS: isFinite(snap.maxSamplePeak) ? Number(snap.maxSamplePeak.toFixed(2)) : null,
      plrLU: isFinite(snap.plr) ? Number(snap.plr.toFixed(2)) : null,
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
  if (!isFinite(sec) || sec < 0) return '00:00.0';
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

function resetMeters() {
  engine.reset();
  liveHistory = [];
  if (currentMode === 'file' && !fileIsPlaying) {
    offlineAnalysisResult = null;
    loadedAudioBuffer = null;
    loadedFileName = '';
    btnPlayPause.disabled = true;
    btnStopFile.disabled = true;
    seekSlider.disabled = true;
    setStatus('standby', 'STANDBY');
    fileMetaInfo.textContent = '초기화됨 — 파일을 드롭하거나 상단에서 입력을 선택하세요';
  }
}

// Event bindings
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

windowSelector.addEventListener('click', (e) => {
  const btn = e.target.closest('.seg-btn');
  if (!btn) return;
  timeWindowSec = Number(btn.dataset.sec);
  document.querySelectorAll('#windowSelector .seg-btn').forEach((b) => b.classList.remove('active'));
  btn.classList.add('active');
});

btnPauseGraph.addEventListener('click', () => {
  graphPaused = !graphPaused;
  btnPauseGraph.classList.toggle('active', graphPaused);
  btnPauseGraph.textContent = graphPaused ? '재개' : '일시정지';
});

// File transport controls
btnPlayPause.addEventListener('click', () => {
  if (!loadedAudioBuffer) return;
  if (fileIsPlaying) pauseFilePlayback();
  else startFilePlayback(filePauseOffsetSec);
});

btnStopFile.addEventListener('click', stopFilePlayback);

seekSlider.addEventListener('input', (e) => {
  if (!loadedAudioBuffer) return;
  const targetSec = (Number(e.target.value) / 1000) * loadedAudioBuffer.duration;
  filePauseOffsetSec = targetSec;
  curTimeEl.textContent = formatTime(targetSec);
  if (fileIsPlaying) {
    startFilePlayback(targetSec);
  }
});

// Canvas hover & click-to-seek
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
  const padLeft = 42;
  const padRight = 14;
  const plotW = Math.max(10, rect.width - padLeft - padRight);
  const clickX = e.clientX - rect.left;
  if (clickX < padLeft || clickX > padLeft + plotW) return;

  const ratio = (clickX - padLeft) / plotW;
  const targetSec = ratio * loadedAudioBuffer.duration;
  filePauseOffsetSec = targetSec;
  seekSlider.value = Math.round(ratio * 1000);
  curTimeEl.textContent = formatTime(targetSec);
  if (fileIsPlaying) {
    startFilePlayback(targetSec);
  }
});

// Drag & Drop handlers on whole window
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

// Keyboard shortcuts
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

if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch((err) => {
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
