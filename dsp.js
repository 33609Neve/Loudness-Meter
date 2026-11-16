/**
 * ITU-R BS.1770-4 / EBU R128 Loudness, True Peak & Stereo Phase DSP Engine
 * Implements:
 * - K-Weighting Filter (Stage 1 High-Shelf + Stage 2 RLB High-Pass) for arbitrary sample rates
 * - 4x Oversampled True Peak Detector (ITU-R BS.1770-4 4-phase 12-tap FIR interpolation)
 * - Momentary (400ms), Short-Term (3000ms), Integrated (Gated -70 LUFS abs / -10 LU rel)
 * - Loudness Range (LRA - EBU Tech 3342: -70 LUFS abs / -20 LU rel, 10th to 95th percentile)
 * - Windowed PSR (Peak to Short-Term Loudness Ratio), PLR, and Normalized Stereo Phase Correlation
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

const TRUE_PEAK_PHASES = [
  new Float64Array([
    0.001708984375, 0.010986328125, -0.0196533203125, 0.033203125,
    -0.0594482421875, 0.1373291015625, 0.97216796875, -0.102294921875,
    0.047607421875, -0.026611328125, 0.014892578125, -0.00830078125,
  ]),
  new Float64Array([
    -0.0291748046875, 0.029296875, -0.0517578125, 0.089111328125,
    -0.16650390625, 0.465087890625, 0.77978515625, -0.2003173828125,
    0.0908203125, -0.0491943359375, 0.026123046875, -0.0125732421875,
  ]),
  new Float64Array([
    -0.0189208984375, 0.0330810546875, -0.0582275390625, 0.1015625,
    -0.2003173828125, 0.77978515625, 0.465087890625, -0.16650390625,
    0.089111328125, -0.0517578125, 0.029296875, -0.0291748046875,
  ]),
  new Float64Array([
    -0.00830078125, 0.014892578125, -0.026611328125, 0.047607421875,
    -0.102294921875, 0.97216796875, 0.1373291015625, -0.0594482421875,
    0.033203125, -0.0196533203125, 0.010986328125, 0.001708984375,
  ]),
];

class ChannelState {
  constructor() {
    this.s1_x1 = 0;
    this.s1_x2 = 0;
    this.s1_y1 = 0;
    this.s1_y2 = 0;
    this.s2_x1 = 0;
    this.s2_x2 = 0;
    this.s2_y1 = 0;
    this.s2_y2 = 0;
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

// 50ms sub-blocks (20 Hz internal resolution):
// Momentary (400ms) = 8 sub-blocks, Short-Term (3000ms) = 60 sub-blocks
const SUB_BLOCKS_PER_SEC = 20;
const MOMENTARY_SUB_BLOCKS = 8;   // 400 ms
const SHORT_TERM_SUB_BLOCKS = 60; // 3000 ms

class LoudnessMeterEngine {
  constructor(sampleRate = 48000, numChannels = 2) {
    this.setConfig(sampleRate, numChannels);
  }

  setConfig(sampleRate, numChannels = 2) {
    this.sampleRate = sampleRate;
    this.numChannels = Math.min(Math.max(numChannels, 1), 2);
    this.coeffs = computeKWeightingCoeffs(sampleRate);
    this.channels = Array.from({ length: this.numChannels }, () => new ChannelState());

    this.subBlockSize = Math.max(1, Math.round(sampleRate / SUB_BLOCKS_PER_SEC));
    this.reset();
  }

  reset() {
    for (const ch of this.channels) ch.reset();

    this.currentSubBlockSamples = 0;
    this.currentSubBlockEnergy = new Float64Array(this.numChannels);
    this.currentSubBlockPeak = 0;
    this.currentSubBlockLR = 0;
    this.currentSubBlockLL = 0;
    this.currentSubBlockRR = 0;

    // Circular buffer of recent 60 sub-blocks (3.0 seconds) of mean-square energy
    this.subBlockRing = new Float64Array(SHORT_TERM_SUB_BLOCKS);
    this.subBlockCount = 0;
    this.subBlockWriteIdx = 0;
    this.psrPeakLinear = 0;

    // Gating & LRA history (sampled every 100ms = every 2 sub-blocks per BS.1770-4 / Tech 3342)
    this.momentaryEnergies = [];
    this.shortTermEnergies = [];

    // Live readout values
    this.momentaryLUFS = -Infinity;
    this.momentaryMaxLUFS = -Infinity;
    this.shortTermLUFS = -Infinity;
    this.shortTermMaxLUFS = -Infinity;
    this.integratedLUFS = -Infinity;
    this.lra = 0.0;
    this.psrLU = -Infinity;

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

    this.subBlocksTotal = 0;
    this.blocksSinceGateUpdate = 0;
    this.totalSamplesProcessed = 0;
  }

  /**
   * Reset a single specific metric and immediately begin re-measuring it
   */
  resetMetric(metricKey) {
    switch (metricKey) {
      case 'shortTerm':
        this.subBlockRing.fill(0);
        this.subBlockCount = 0;
        this.subBlockWriteIdx = 0;
        this.shortTermLUFS = -Infinity;
        break;
      case 'integrated':
        this.momentaryEnergies = [];
        this.integratedLUFS = -Infinity;
        break;
      case 'lra':
        this.shortTermEnergies = [];
        this.lra = 0.0;
        break;
      case 'psr':
        this.psrPeakLinear = 0;
        this.psrLU = -Infinity;
        break;
      case 'plr':
        this.maxTruePeak = 0;
        this.maxTruePeakCh.fill(0);
        this.momentaryEnergies = [];
        this.integratedLUFS = -Infinity;
        break;
      case 'momentaryMax':
        this.momentaryMaxLUFS = isFinite(this.momentaryLUFS) ? this.momentaryLUFS : -Infinity;
        break;
      case 'shortTermMax':
        this.shortTermMaxLUFS = isFinite(this.shortTermLUFS) ? this.shortTermLUFS : -Infinity;
        break;
      case 'truePeakMax':
        this.maxTruePeak = 0;
        this.maxSamplePeak = 0;
        this.maxTruePeakCh.fill(0);
        this.clipCount = 0;
        break;
      case 'phase':
        this.minCorrelation = this.correlation;
        break;
      default:
        break;
    }
  }

  process(channelBuffers) {
    const numCh = Math.min(channelBuffers.length, this.numChannels);
    if (numCh === 0) return;
    const length = channelBuffers[0].length;
    if (length === 0) return;

    const { stage1, stage2 } = this.coeffs;

    const blockSamplePeak = new Float64Array(this.numChannels);
    const blockTruePeak = new Float64Array(this.numChannels);

    for (let i = 0; i < length; i++) {
      const sampleL = channelBuffers[0][i];
      const sampleR = numCh > 1 ? channelBuffers[1][i] : sampleL;
      this.currentSubBlockLR += sampleL * sampleR;
      this.currentSubBlockLL += sampleL * sampleL;
      this.currentSubBlockRR += sampleR * sampleR;

      for (let ch = 0; ch < numCh; ch++) {
        const x = channelBuffers[ch][i];
        const state = this.channels[ch];

        // 1. Sample Peak
        const absX = Math.abs(x);
        if (absX > blockSamplePeak[ch]) blockSamplePeak[ch] = absX;
        if (absX > this.maxSamplePeak) this.maxSamplePeak = absX;

        // 2. 4x Oversampled True Peak
        state.tpHistory[state.tpIndex] = x;
        let sampleTP = absX;
        if (absX > 0.01 || absX >= blockTruePeak[ch] * 0.5) {
          for (let p = 0; p < 4; p++) {
            const taps = TRUE_PEAK_PHASES[p];
            let acc = 0;
            let idx = state.tpIndex;
            for (let t = 0; t < 12; t++) {
              acc += taps[t] * state.tpHistory[idx];
              idx = idx === 0 ? 11 : idx - 1;
            }
            const absAcc = Math.abs(acc);
            if (absAcc > sampleTP) sampleTP = absAcc;
          }
          if (sampleTP > this.maxTruePeakCh[ch]) this.maxTruePeakCh[ch] = sampleTP;
          if (sampleTP > this.maxTruePeak) this.maxTruePeak = sampleTP;
          if (sampleTP > 1.0) this.clipCount++;
        }
        if (sampleTP > blockTruePeak[ch]) blockTruePeak[ch] = sampleTP;
        if (sampleTP > this.currentSubBlockPeak) this.currentSubBlockPeak = sampleTP;
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

    // Smooth decay for instantaneous peak bar meters
    for (let ch = 0; ch < this.numChannels; ch++) {
      this.samplePeakCh[ch] = Math.max(blockSamplePeak[ch], this.samplePeakCh[ch] * 0.92);
      this.truePeakCh[ch] = Math.max(blockTruePeak[ch], this.truePeakCh[ch] * 0.92);
    }
  }

  _commitSubBlock() {
    const N = this.currentSubBlockSamples || 1;
    let totalSubBlockMeanSquare = 0;
    for (let ch = 0; ch < this.numChannels; ch++) {
      totalSubBlockMeanSquare += this.currentSubBlockEnergy[ch] / N;
      this.currentSubBlockEnergy[ch] = 0;
    }
    this.currentSubBlockSamples = 0;

    // Normalized per-sub-block Stereo Phase Correlation (amplitude-independent above -70 dBFS)
    const meanLL = this.currentSubBlockLL / N;
    const meanRR = this.currentSubBlockRR / N;
    const denom = Math.sqrt(this.currentSubBlockLL * this.currentSubBlockRR);
    if (meanLL + meanRR > 1e-7 && denom > 1e-10) {
      const subCorr = Math.max(-1, Math.min(1, this.currentSubBlockLR / denom));
      this.correlation = this.correlation * 0.72 + subCorr * 0.28;
      if (this.correlation < this.minCorrelation) {
        this.minCorrelation = this.correlation;
      }
      this.balance = (meanRR - meanLL) / (meanLL + meanRR);
    }
    this.currentSubBlockLR = 0;
    this.currentSubBlockLL = 0;
    this.currentSubBlockRR = 0;

    this.subBlockRing[this.subBlockWriteIdx] = totalSubBlockMeanSquare;
    // Smooth exponential peak follower (fast attack, slow 3s release) so PSR never drops off a cliff
    if (this.currentSubBlockPeak >= this.psrPeakLinear) {
      this.psrPeakLinear = this.psrPeakLinear * 0.25 + this.currentSubBlockPeak * 0.75;
    } else {
      this.psrPeakLinear *= 0.975;
    }
    this.currentSubBlockPeak = 0;

    this.subBlockWriteIdx = (this.subBlockWriteIdx + 1) % SHORT_TERM_SUB_BLOCKS;
    if (this.subBlockCount < SHORT_TERM_SUB_BLOCKS) this.subBlockCount++;
    this.subBlocksTotal++;

    // 1. Momentary Loudness (400ms = 8 sub-blocks of 50ms)
    const mCount = Math.min(MOMENTARY_SUB_BLOCKS, this.subBlockCount);
    let mEnergySum = 0;
    for (let k = 0; k < mCount; k++) {
      const idx = (this.subBlockWriteIdx - 1 - k + SHORT_TERM_SUB_BLOCKS) % SHORT_TERM_SUB_BLOCKS;
      mEnergySum += this.subBlockRing[idx];
    }
    const mMeanEnergy = mEnergySum / mCount;
    this.momentaryLUFS = energyToLUFS(mMeanEnergy);
    if (this.subBlockCount >= MOMENTARY_SUB_BLOCKS && this.momentaryLUFS > this.momentaryMaxLUFS && isFinite(this.momentaryLUFS)) {
      this.momentaryMaxLUFS = this.momentaryLUFS;
    }

    // Per ITU-R BS.1770-4, gating blocks step every 100ms (every 2nd 50ms sub-block)
    const is100msStep = this.subBlocksTotal % 2 === 0;
    if (is100msStep && this.subBlockCount >= MOMENTARY_SUB_BLOCKS) {
      if (mMeanEnergy > 1.1719255e-7) {
        this.momentaryEnergies.push(mMeanEnergy);
      }
    }

    // 2. Short-Term Loudness (3000ms = 60 sub-blocks of 50ms) & Smooth PSR
    const stCount = this.subBlockCount;
    let stEnergySum = 0;
    for (let k = 0; k < stCount; k++) {
      const idx = (this.subBlockWriteIdx - 1 - k + SHORT_TERM_SUB_BLOCKS) % SHORT_TERM_SUB_BLOCKS;
      stEnergySum += this.subBlockRing[idx];
    }
    const stMeanEnergy = stEnergySum / stCount;
    this.shortTermLUFS = energyToLUFS(stMeanEnergy);

    const stWindowPeakDB = linearToDB(this.psrPeakLinear);
    if (isFinite(stWindowPeakDB) && isFinite(this.shortTermLUFS) && this.shortTermLUFS > -70) {
      const rawPSR = Math.max(0, stWindowPeakDB - this.shortTermLUFS);
      this.psrLU = isFinite(this.psrLU) ? this.psrLU * 0.85 + rawPSR * 0.15 : rawPSR;
    }

    if (this.subBlockCount >= 20 && this.shortTermLUFS > this.shortTermMaxLUFS && isFinite(this.shortTermLUFS)) {
      this.shortTermMaxLUFS = this.shortTermLUFS;
    }

    if (is100msStep && this.subBlockCount >= SHORT_TERM_SUB_BLOCKS) {
      if (stMeanEnergy > 1.1719255e-7) {
        this.shortTermEnergies.push(stMeanEnergy);
      }
    }

    // Update Integrated LUFS and LRA every 200ms (4 sub-blocks)
    this.blocksSinceGateUpdate++;
    if (this.blocksSinceGateUpdate >= 4) {
      this.blocksSinceGateUpdate = 0;
      this._updateIntegratedAndLRA();
    }
  }

  _updateIntegratedAndLRA() {
    // 1. Gated Integrated Loudness (BS.1770-4)
    const mBlocks = this.momentaryEnergies;
    const nM = mBlocks.length;
    if (nM > 0) {
      let absSum = 0;
      for (let i = 0; i < nM; i++) absSum += mBlocks[i];
      const gammaAbsMean = absSum / nM;
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
    if (nST >= 5) {
      let stAbsSum = 0;
      for (let i = 0; i < nST; i++) stAbsSum += stBlocks[i];
      const stAbsMean = stAbsSum / nST;
      const lraRelThreshold = stAbsMean * 0.01;

      const gatedLufs = [];
      for (let i = 0; i < nST; i++) {
        if (stBlocks[i] >= lraRelThreshold) {
          gatedLufs.push(energyToLUFS(stBlocks[i]));
        }
      }

      if (gatedLufs.length >= 5) {
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
        ? Math.max(0, maxTpDB - this.integratedLUFS)
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
      psr: this.psrLU,
      correlation: this.correlation,
      minCorrelation: this.minCorrelation,
      balance: this.balance,
      clipCount: this.clipCount,
      durationSec: this.totalSamplesProcessed / this.sampleRate,
    };
  }
}

function analyzeAudioBufferOffline(audioBuffer) {
  const sampleRate = audioBuffer.sampleRate;
  const engine = new LoudnessMeterEngine(sampleRate, 2);

  const ch0 = audioBuffer.getChannelData(0);
  const ch1 = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : ch0;
  const channels = [ch0, ch1];

  const totalSamples = audioBuffer.length;
  // Step every 100ms (2 sub-blocks) for timeline points
  const stepSamples = engine.subBlockSize * 2;
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
      psr: snap.psr,
      correlation: snap.correlation,
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
