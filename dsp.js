/**
 * ITU-R BS.1770-4 / EBU R128 Loudness & True Peak DSP Engine
 * Implements:
 * - K-Weighting Filter (Stage 1 High-Shelf + Stage 2 RLB High-Pass) for arbitrary sample rates
 * - 4x Oversampled True Peak Detector (ITU-R BS.1770-4 4-phase 12-tap FIR interpolation)
 * - Momentary (400ms), Short-Term (3000ms), Integrated (Gated -70 LUFS abs / -10 LU rel)
 * - Loudness Range (LRA - EBU Tech 3342: -70 LUFS abs / -20 LU rel, 10th to 95th percentile)
 * - Fast Offline File Analyzer for instant full-track loudness profile
 */

(function (global) {
const STREAMING_PRESETS = {
  spotify: { name: 'Spotify', targetLUFS: -14.0, maxTruePeak: -1.0, tolerance: 1.0, category: 'Streaming' },
  youtube: { name: 'YouTube', targetLUFS: -14.0, maxTruePeak: -1.0, tolerance: 1.0, category: 'Streaming' },
  apple: { name: 'Apple Music', targetLUFS: -16.0, maxTruePeak: -1.0, tolerance: 1.0, category: 'Streaming' },
  tidal: { name: 'Tidal / Amazon', targetLUFS: -14.0, maxTruePeak: -1.0, tolerance: 1.0, category: 'Streaming' },
  soundcloud: { name: 'SoundCloud', targetLUFS: -14.0, maxTruePeak: -1.0, tolerance: 1.5, category: 'Streaming' },
  ebu: { name: 'EBU R128 (Broadcast)', targetLUFS: -23.0, maxTruePeak: -1.0, tolerance: 0.5, category: 'Broadcast' },
  atsc: { name: 'ATSC A/85 (US TV)', targetLUFS: -24.0, maxTruePeak: -2.0, tolerance: 1.0, category: 'Broadcast' },
  netflix: { name: 'Netflix', targetLUFS: -27.0, maxTruePeak: -2.0, tolerance: 1.0, category: 'Broadcast' },
  podcast: { name: 'Podcast (Stereo)', targetLUFS: -16.0, maxTruePeak: -1.0, tolerance: 1.0, category: 'Voice' },
  club: { name: 'Club / Loud Master', targetLUFS: -8.0, maxTruePeak: -0.3, tolerance: 1.5, category: 'Mastering' },
};

/**
 * Compute K-Weighting Biquad Filter Coefficients for any sample rate
 * Based on ITU-R BS.1770-4 analytical bilinear transform definitions
 */
function computeKWeightingCoeffs(sampleRate) {
  // Stage 1: High-shelf pre-filter (acoustic effects of the head)
  const db = 3.999843853973347;
  const f0_1 = 1681.974450955533;
  const Q_1 = 0.7071752369554196;
  const K_1 = Math.tan((Math.PI * f0_1) / sampleRate);
  const Vh = Math.pow(10.0, db / 20.0);
  const Vb = Math.pow(Vh, 0.4996667741545416);

  const a0_1 = 1.0 + K_1 / Q_1 + K_1 * K_1;
  const stage1 = {
    b0: (Vh + (Vb * K_1) / Q_1 + K_1 * K_1) / a0_1,
    b1: (2.0 * (K_1 * K_1 - Vh)) / a0_1,
    b2: (Vh - (Vb * K_1) / Q_1 + K_1 * K_1) / a0_1,
    a1: (2.0 * (K_1 * K_1 - 1.0)) / a0_1,
    a2: (1.0 - K_1 / Q_1 + K_1 * K_1) / a0_1,
  };

  // Stage 2: RLB High-pass filter
  const f0_2 = 38.13547087602444;
  const Q_2 = 0.5003270373238773;
  const K_2 = Math.tan((Math.PI * f0_2) / sampleRate);
  const a0_2 = 1.0 + K_2 / Q_2 + K_2 * K_2;
  const stage2 = {
    b0: 1.0 / a0_2,
    b1: -2.0 / a0_2,
    b2: 1.0 / a0_2,
    a1: (2.0 * (K_2 * K_2 - 1.0)) / a0_2,
    a2: (1.0 - K_2 / Q_2 + K_2 * K_2) / a0_2,
  };

  return { stage1, stage2 };
}

/**
 * ITU-R BS.1770-4 Table 3: 48-tap 4-phase FIR filter coefficients for 4x True Peak oversampling
 */
const TRUE_PEAK_PHASES = [
  // Phase 0
  new Float64Array([
    0.001708984375, 0.010986328125, -0.0196533203125, 0.033203125,
    -0.0594482421875, 0.1373291015625, 0.97216796875, -0.102294921875,
    0.047607421875, -0.026611328125, 0.014892578125, -0.00830078125,
  ]),
  // Phase 1
  new Float64Array([
    -0.0291748046875, 0.029296875, -0.0517578125, 0.089111328125,
    -0.16650390625, 0.465087890625, 0.77978515625, -0.2003173828125,
    0.0908203125, -0.0491943359375, 0.026123046875, -0.0125732421875,
  ]),
  // Phase 2
  new Float64Array([
    -0.0189208984375, 0.0330810546875, -0.0582275390625, 0.1015625,
    -0.2003173828125, 0.77978515625, 0.465087890625, -0.16650390625,
    0.089111328125, -0.0517578125, 0.029296875, -0.0291748046875,
  ]),
  // Phase 3
  new Float64Array([
    -0.00830078125, 0.014892578125, -0.026611328125, 0.047607421875,
    -0.102294921875, 0.97216796875, 0.1373291015625, -0.0594482421875,
    0.033203125, -0.0196533203125, 0.010986328125, 0.001708984375,
  ]),
];

class ChannelState {
  constructor() {
    // Stage 1 biquad state
    this.s1_x1 = 0;
    this.s1_x2 = 0;
    this.s1_y1 = 0;
    this.s1_y2 = 0;
    // Stage 2 biquad state
    this.s2_x1 = 0;
    this.s2_x2 = 0;
    this.s2_y1 = 0;
    this.s2_y2 = 0;
    // 12-sample FIR delay line for 4x True Peak interpolation
    this.tpHistory = new Float64Array(12);
    this.tpIndex = 0;
  }

  reset() {
    this.s1_x1 = this.s1_x2 = this.s1_y1 = this.s1_y2 = 0;
    this.s2_x1 = this.s2_x2 = this.s2_y1 = this.s2_y2 = 0;
    this.tpHistory.fill(0);
    this.tpIndex = 0;
  }
}

class LoudnessMeterEngine {
  constructor(sampleRate = 48000, numChannels = 2) {
    this.setConfig(sampleRate, numChannels);
  }

  setConfig(sampleRate, numChannels = 2) {
    this.sampleRate = sampleRate;
    this.numChannels = Math.min(Math.max(numChannels, 1), 2);
    this.coeffs = computeKWeightingCoeffs(sampleRate);
    this.channels = Array.from({ length: this.numChannels }, () => new ChannelState());

    // 100ms sub-block length (10 Hz step per BS.1770-4)
    this.subBlockSize = Math.max(1, Math.round(sampleRate * 0.1));
    this.reset();
  }

  reset() {
    for (const ch of this.channels) ch.reset();

    this.currentSubBlockSamples = 0;
    this.currentSubBlockEnergy = new Float64Array(this.numChannels);

    // Circular buffer of recent 30 sub-blocks (3.0 seconds) of mean-square energy
    this.subBlockRing = new Float64Array(30);
    this.subBlockCount = 0;
    this.subBlockWriteIdx = 0;

    // Gating & LRA history
    // Each 400ms block (every 100ms after first 400ms) stores combined energy z_j
    this.momentaryEnergies = [];
    // Each 3000ms block (every 100ms after first 3.0s) stores combined energy for LRA
    this.shortTermEnergies = [];

    // Live readout values
    this.momentaryLUFS = -Infinity;
    this.momentaryMaxLUFS = -Infinity;
    this.shortTermLUFS = -Infinity;
    this.shortTermMaxLUFS = -Infinity;
    this.integratedLUFS = -Infinity;
    this.lra = 0.0;

    // Peaks (linear amplitude, converted to dB on read)
    this.samplePeakCh = new Float64Array(this.numChannels);
    this.truePeakCh = new Float64Array(this.numChannels);
    this.maxTruePeakCh = new Float64Array(this.numChannels);
    this.maxSamplePeak = 0;
    this.maxTruePeak = 0;
    this.clipCount = 0;

    // Stereo Phase Correlation (-1 to +1) & L/R Balance
    this.correlation = 1.0;
    this.minCorrelation = 1.0;
    this.balance = 0.0;
    this._corrSumLR = 0;
    this._corrSumLL = 0;
    this._corrSumRR = 0;

    // Cached gated calculation throttling
    this.blocksSinceGateUpdate = 0;
    this.totalSamplesProcessed = 0;
  }

  /**
   * Process a chunk of audio channels: Array of Float32Array [left, right]
   */
  process(channelBuffers) {
    const numCh = Math.min(channelBuffers.length, this.numChannels);
    if (numCh === 0) return;
    const length = channelBuffers[0].length;
    if (length === 0) return;

    const { stage1, stage2 } = this.coeffs;

    // Per-block instantaneous peaks & phase correlation accumulators
    const blockSamplePeak = new Float64Array(this.numChannels);
    const blockTruePeak = new Float64Array(this.numChannels);
    let chunkLR = 0;
    let chunkLL = 0;
    let chunkRR = 0;

    for (let i = 0; i < length; i++) {
      const sampleL = channelBuffers[0][i];
      const sampleR = numCh > 1 ? channelBuffers[1][i] : sampleL;
      chunkLR += sampleL * sampleR;
      chunkLL += sampleL * sampleL;
      chunkRR += sampleR * sampleR;

      for (let ch = 0; ch < numCh; ch++) {
        const x = channelBuffers[ch][i];
        const state = this.channels[ch];

        // 1. Sample Peak
        const absX = Math.abs(x);
        if (absX > blockSamplePeak[ch]) blockSamplePeak[ch] = absX;
        if (absX > this.maxSamplePeak) this.maxSamplePeak = absX;

        // 2. 4x Oversampled True Peak
        state.tpHistory[state.tpIndex] = x;
        // Fast check: only run full 4-phase convolution when signal is significant (> -40 dBFS) or near current block peak
        if (absX > 0.01 || absX >= blockTruePeak[ch] * 0.5) {
          let maxPhaseAmp = absX;
          for (let p = 0; p < 4; p++) {
            const taps = TRUE_PEAK_PHASES[p];
            let acc = 0;
            let idx = state.tpIndex;
            for (let t = 0; t < 12; t++) {
              acc += taps[t] * state.tpHistory[idx];
              idx = idx === 0 ? 11 : idx - 1;
            }
            const absAcc = Math.abs(acc);
            if (absAcc > maxPhaseAmp) maxPhaseAmp = absAcc;
          }
          if (maxPhaseAmp > blockTruePeak[ch]) blockTruePeak[ch] = maxPhaseAmp;
          if (maxPhaseAmp > this.maxTruePeakCh[ch]) this.maxTruePeakCh[ch] = maxPhaseAmp;
          if (maxPhaseAmp > this.maxTruePeak) this.maxTruePeak = maxPhaseAmp;
          if (maxPhaseAmp > 1.0) this.clipCount++;
        } else {
          if (absX > blockTruePeak[ch]) blockTruePeak[ch] = absX;
        }
        state.tpIndex = (state.tpIndex + 1) % 12;

        // 3. K-Weighting Stage 1 (High-Shelf)
        const y1 =
          stage1.b0 * x +
          stage1.b1 * state.s1_x1 +
          stage1.b2 * state.s1_x2 -
          stage1.a1 * state.s1_y1 -
          stage1.a2 * state.s1_y2;
        state.s1_x2 = state.s1_x1;
        state.s1_x1 = x;
        state.s1_y2 = state.s1_y1;
        state.s1_y1 = y1;

        // 4. K-Weighting Stage 2 (RLB High-Pass)
        const y2 =
          stage2.b0 * y1 +
          stage2.b1 * state.s2_x1 +
          stage2.b2 * state.s2_x2 -
          stage2.a1 * state.s2_y1 -
          stage2.a2 * state.s2_y2;
        state.s2_x2 = state.s2_x1;
        state.s2_x1 = y1;
        state.s2_y2 = state.s2_y1;
        state.s2_y1 = y2;

        this.currentSubBlockEnergy[ch] += y2 * y2;
      }

      // If input is mono but engine is stereo, duplicate Ch0 into Ch1
      if (numCh === 1 && this.numChannels === 2) {
        blockSamplePeak[1] = blockSamplePeak[0];
        blockTruePeak[1] = blockTruePeak[0];
        this.maxTruePeakCh[1] = this.maxTruePeakCh[0];
        this.currentSubBlockEnergy[1] = this.currentSubBlockEnergy[0];
      }

      this.currentSubBlockSamples++;
      this.totalSamplesProcessed++;

      if (this.currentSubBlockSamples >= this.subBlockSize) {
        this._commitSubBlock();
      }
    }

    // Smooth decay for instantaneous bar meters
    for (let ch = 0; ch < this.numChannels; ch++) {
      this.samplePeakCh[ch] = Math.max(blockSamplePeak[ch], this.samplePeakCh[ch] * 0.86);
      this.truePeakCh[ch] = Math.max(blockTruePeak[ch], this.truePeakCh[ch] * 0.86);
    }

    // Update Stereo Phase Correlation & Balance with exponential smoothing
    this._corrSumLR = this._corrSumLR * 0.78 + chunkLR * 0.22;
    this._corrSumLL = this._corrSumLL * 0.78 + chunkLL * 0.22;
    this._corrSumRR = this._corrSumRR * 0.78 + chunkRR * 0.22;
    const denom = Math.sqrt(this._corrSumLL * this._corrSumRR);
    if (denom > 1e-9) {
      const instCorr = Math.max(-1, Math.min(1, this._corrSumLR / denom));
      this.correlation = instCorr;
      if (instCorr < this.minCorrelation) {
        this.minCorrelation = instCorr;
      }
      const totalLR = this._corrSumLL + this._corrSumRR;
      this.balance = totalLR > 1e-9 ? (this._corrSumRR - this._corrSumLL) / totalLR : 0;
    }
  }

  _commitSubBlock() {
    // Mean square energy across channels (G_i = 1.0 for L/R)
    let totalSubBlockMeanSquare = 0;
    for (let ch = 0; ch < this.numChannels; ch++) {
      totalSubBlockMeanSquare += this.currentSubBlockEnergy[ch] / this.currentSubBlockSamples;
      this.currentSubBlockEnergy[ch] = 0;
    }
    // If mono (1 channel), loudness of a single full-scale sine wave in stereo vs mono:
    // In BS.1770, mono is either routed to L+R or scaled. Standard DAW meters treat mono input as dual-mono or single channel.
    this.currentSubBlockSamples = 0;

    this.subBlockRing[this.subBlockWriteIdx] = totalSubBlockMeanSquare;
    this.subBlockWriteIdx = (this.subBlockWriteIdx + 1) % 30;
    if (this.subBlockCount < 30) this.subBlockCount++;

    // Momentary (400ms = last 4 sub-blocks)
    const mCount = Math.min(4, this.subBlockCount);
    let mEnergySum = 0;
    for (let k = 0; k < mCount; k++) {
      const idx = (this.subBlockWriteIdx - 1 - k + 30) % 30;
      mEnergySum += this.subBlockRing[idx];
    }
    const mMeanEnergy = mEnergySum / mCount;
    this.momentaryLUFS = energyToLUFS(mMeanEnergy);
    if (this.momentaryLUFS > this.momentaryMaxLUFS && isFinite(this.momentaryLUFS)) {
      this.momentaryMaxLUFS = this.momentaryLUFS;
    }

    if (this.subBlockCount >= 4) {
      // Only store blocks above absolute gate (-70 LUFS -> energy > 10^((-70 + 0.691)/10))
      if (mMeanEnergy > 1.1719255e-7) {
        this.momentaryEnergies.push(mMeanEnergy);
      }
    }

    // Short-term (3000ms = last 30 sub-blocks)
    const stCount = this.subBlockCount;
    let stEnergySum = 0;
    for (let k = 0; k < stCount; k++) {
      const idx = (this.subBlockWriteIdx - 1 - k + 30) % 30;
      stEnergySum += this.subBlockRing[idx];
    }
    const stMeanEnergy = stEnergySum / stCount;
    this.shortTermLUFS = energyToLUFS(stMeanEnergy);
    if (this.shortTermLUFS > this.shortTermMaxLUFS && isFinite(this.shortTermLUFS)) {
      this.shortTermMaxLUFS = this.shortTermLUFS;
    }

    if (this.subBlockCount >= 30) {
      if (stMeanEnergy > 1.1719255e-7) {
        this.shortTermEnergies.push(stMeanEnergy);
      }
    }

    // Update Integrated LUFS and LRA every 200ms (2 sub-blocks)
    this.blocksSinceGateUpdate++;
    if (this.blocksSinceGateUpdate >= 2) {
      this.blocksSinceGateUpdate = 0;
      this._updateIntegratedAndLRA();
    }
  }

  _updateIntegratedAndLRA() {
    // 1. Gated Integrated Loudness (BS.1770-4)
    const mBlocks = this.momentaryEnergies;
    const nM = mBlocks.length;
    if (nM > 0) {
      // All entries in mBlocks already passed the -70 LUFS absolute gate (1.1719255e-7)
      let absSum = 0;
      for (let i = 0; i < nM; i++) absSum += mBlocks[i];
      const gammaAbsMean = absSum / nM;
      // Relative threshold is -10 LU below absolute-gated mean -> multiply energy by 0.1
      const relThresholdEnergy = gammaAbsMean * 0.1;

      let relSum = 0;
      let relCount = 0;
      for (let i = 0; i < nM; i++) {
        if (mBlocks[i] >= relThresholdEnergy) {
          relSum += mBlocks[i];
          relCount++;
        }
      }
      if (relCount > 0) {
        this.integratedLUFS = energyToLUFS(relSum / relCount);
      }
    }

    // 2. Loudness Range (LRA - EBU Tech 3342)
    const stBlocks = this.shortTermEnergies;
    const nST = stBlocks.length;
    if (nST >= 2) {
      let stAbsSum = 0;
      for (let i = 0; i < nST; i++) stAbsSum += stBlocks[i];
      const stAbsMean = stAbsSum / nST;
      // Relative gate for LRA is -20 LU -> energy * 0.01
      const lraRelThreshold = stAbsMean * 0.01;

      const gatedLufs = [];
      for (let i = 0; i < nST; i++) {
        if (stBlocks[i] >= lraRelThreshold) {
          gatedLufs.push(energyToLUFS(stBlocks[i]));
        }
      }

      if (gatedLufs.length >= 2) {
        gatedLufs.sort((a, b) => a - b);
        const p10Idx = Math.floor((gatedLufs.length - 1) * 0.1);
        const p95Idx = Math.floor((gatedLufs.length - 1) * 0.95);
        this.lra = Math.max(0, gatedLufs[p95Idx] - gatedLufs[p10Idx]);
      }
    }
  }

  getSnapshot() {
    const tpLeftDB = linearToDB(this.truePeakCh[0] || 0);
    const tpRightDB = linearToDB(this.truePeakCh[1] ?? this.truePeakCh[0] ?? 0);
    const maxTpDB = linearToDB(this.maxTruePeak);
    const maxSpDB = linearToDB(this.maxSamplePeak);
    const plr =
      isFinite(maxTpDB) && isFinite(this.integratedLUFS)
        ? maxTpDB - this.integratedLUFS
        : -Infinity;
    const currentPeakDB = Math.max(tpLeftDB, tpRightDB);
    const psr =
      isFinite(currentPeakDB) && isFinite(this.shortTermLUFS)
        ? currentPeakDB - this.shortTermLUFS
        : -Infinity;

    return {
      momentary: this.momentaryLUFS,
      momentaryMax: this.momentaryMaxLUFS,
      shortTerm: this.shortTermLUFS,
      shortTermMax: this.shortTermMaxLUFS,
      integrated: this.integratedLUFS,
      lra: this.lra,
      truePeakL: tpLeftDB,
      truePeakR: tpRightDB,
      maxTruePeak: maxTpDB,
      maxSamplePeak: maxSpDB,
      plr,
      psr,
      correlation: this.correlation,
      minCorrelation: this.minCorrelation,
      balance: this.balance,
      clipCount: this.clipCount,
      durationSec: this.totalSamplesProcessed / this.sampleRate,
    };
  }
}

/**
 * Offline Full-File Analyzer
 * Processes an AudioBuffer and returns both summary metrics and a high-resolution timeline
 * for plotting the entire song's loudness contour immediately on file drop.
 */
function analyzeAudioBufferOffline(audioBuffer) {
  const sampleRate = audioBuffer.sampleRate;
  const engine = new LoudnessMeterEngine(sampleRate, 2);

  const ch0 = audioBuffer.getChannelData(0);
  const ch1 = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : ch0;
  const channels = [ch0, ch1];

  const totalSamples = audioBuffer.length;
  const stepSamples = engine.subBlockSize; // 100ms chunks
  const timeline = [];

  for (let offset = 0; offset < totalSamples; offset += stepSamples) {
    const end = Math.min(offset + stepSamples, totalSamples);
    const sliceCh = channels.map((ch) => ch.subarray(offset, end));
    engine.process(sliceCh);

    const snap = engine.getSnapshot();
    timeline.push({
      timeSec: end / sampleRate,
      momentary: snap.momentary,
      shortTerm: snap.shortTerm,
      integrated: snap.integrated,
      truePeakL: snap.truePeakL,
      truePeakR: snap.truePeakR,
    });
  }

  engine._updateIntegratedAndLRA();
  return {
    summary: engine.getSnapshot(),
    timeline,
    durationSec: audioBuffer.duration,
    sampleRate,
    numChannels: audioBuffer.numberOfChannels,
  };
}

function energyToLUFS(energy) {
  if (energy <= 1e-12) return -Infinity;
  return -0.691 + 10.0 * Math.log10(energy);
}

function linearToDB(amp) {
  if (amp <= 1e-6) return -Infinity;
  return 20.0 * Math.log10(amp);
}

function formatLUFS(val, decimals = 1) {
  if (!isFinite(val) || val < -90) return '-∞';
  return val.toFixed(decimals);
}

function formatDB(val, decimals = 1) {
  if (!isFinite(val) || val < -90) return '-∞';
  return (val > 0 ? '+' : '') + val.toFixed(decimals);
}

global.ULM_DSP = {
  STREAMING_PRESETS,
  computeKWeightingCoeffs,
  LoudnessMeterEngine,
  analyzeAudioBufferOffline,
  energyToLUFS,
  linearToDB,
  formatLUFS,
  formatDB,
};
})(window);
