// Reliable microphone capture with stereo input and mono WAV output.

export function encodeWav(chunks, sampleRate) {
  let n = 0;
  for (const c of chunks) n += c.length;

  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);

  const w = (o, s) => {
    for (let i = 0; i < s.length; i++) {
      v.setUint8(o + i, s.charCodeAt(i));
    }
  };

  w(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, n * 2, true);

  let offset = 44;

  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i++) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));

      v.setInt16(
        offset,
        sample < 0 ? sample * 32768 : sample * 32767,
        true
      );

      offset += 2;
    }
  }

  return new Blob([buf], { type: "audio/wav" });
}

export class Recorder {
  constructor({ maxSeconds = 120, onLevel, onTick, onAutoStop }) {
    Object.assign(this, {
      maxSeconds,
      onLevel,
      onTick,
      onAutoStop
    });

    this.chunks = [];
    this.samples = 0;
    this.active = false;
    this.autoStopped = false;
  }

  static supported() {
    return !!(
      navigator.mediaDevices &&
      navigator.mediaDevices.getUserMedia &&
      (window.AudioContext || window.webkitAudioContext)
    );
  }

  async start() {
    if (!Recorder.supported()) {
      throw new Error(
        "This browser cannot record audio. Upload a file instead."
      );
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: { ideal: 2 }
        }
      });
    } catch (e) {
      const messages = {
        NotAllowedError:
          "Microphone access was blocked. Allow microphone access in browser settings.",
        NotFoundError:
          "No microphone was found.",
        NotReadableError:
          "The microphone is being used by another application."
      };

      throw new Error(
        messages[e.name] || `Microphone error: ${e.message}`
      );
    }

    const track = this.stream.getAudioTracks()[0];

    console.log("MICROPHONE DEVICE:", track.getSettings());

    const AC = window.AudioContext || window.webkitAudioContext;

    this.ctx = new AC();

    if (this.ctx.state === "suspended") {
      await this.ctx.resume();
    }

    this.source = this.ctx.createMediaStreamSource(this.stream);

    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 1024;

    this.source.connect(this.analyser);

    // Capture two input channels.
    this.processor = this.ctx.createScriptProcessor(4096, 2, 1);

    this.silentGain = this.ctx.createGain();
    this.silentGain.gain.value = 0;

    this.processor.onaudioprocess = (event) => {
      if (!this.active) return;

      const input = event.inputBuffer;
      const channels = input.numberOfChannels;
      const length = input.length;

      const copy = new Float32Array(length);

      // Combine available channels into mono.
      for (let ch = 0; ch < channels; ch++) {
        const channelData = input.getChannelData(ch);

        for (let i = 0; i < length; i++) {
          copy[i] += channelData[i] / channels;
        }
      }

      this.chunks.push(copy);
      this.samples += copy.length;

      const seconds = this.samples / this.ctx.sampleRate;

      this.onTick && this.onTick(seconds);

      if (seconds >= this.maxSeconds && !this.autoStopped) {
        this.autoStopped = true;
        this.onAutoStop && this.onAutoStop();
      }
    };

    this.active = true;

    this.source.connect(this.processor);
    this.processor.connect(this.silentGain);
    this.silentGain.connect(this.ctx.destination);

    // Monitor microphone input.
    const buffer = new Float32Array(this.analyser.fftSize);

    const monitor = () => {
      if (!this.active) return;

      this.analyser.getFloatTimeDomainData(buffer);

      let sum = 0;
      let peak = 0;

      for (const value of buffer) {
        sum += value * value;
        peak = Math.max(peak, Math.abs(value));
      }

      const rms = Math.sqrt(sum / buffer.length);

      this.onLevel && this.onLevel(rms, peak);

      this.raf = requestAnimationFrame(monitor);
    };

    monitor();
  }

  async stop() {
    if (!this.active) {
      throw new Error("Recording is not active.");
    }

    this.active = false;

    cancelAnimationFrame(this.raf);

    const rate = this.ctx.sampleRate;
    const seconds = this.samples / rate;

    // Diagnose actual recorded samples.
    let peak = 0;
    let total = 0;

    for (const chunk of this.chunks) {
      for (const sample of chunk) {
        peak = Math.max(peak, Math.abs(sample));
        total += sample * sample;
      }
    }

    const rms = this.samples
      ? Math.sqrt(total / this.samples)
      : 0;

    console.log("MICROPHONE RECORDING TEST:", {
      samples: this.samples,
      duration: seconds,
      sampleRate: rate,
      peak,
      rms
    });

    this.processor.onaudioprocess = null;

    try {
      this.source.disconnect();
      this.processor.disconnect();
      this.analyser.disconnect();
      this.silentGain.disconnect();
    } catch (e) {
      // Nodes may already be disconnected.
    }

    this.stream.getTracks().forEach(track => track.stop());

    await this.ctx.close();

    const blob = encodeWav(this.chunks, rate);

    this.chunks = [];
    this.samples = 0;

    return { blob, seconds };
  }

  cancel() {
    this.active = false;

    cancelAnimationFrame(this.raf);

    if (this.processor) {
      this.processor.onaudioprocess = null;
    }

    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
    }

    if (this.ctx && this.ctx.state !== "closed") {
      this.ctx.close();
    }

    this.chunks = [];
    this.samples = 0;
  }
}