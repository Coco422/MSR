from __future__ import annotations

import numpy as np
import webrtcvad

from msr.backends.vad.base import VADBackend


class WebRTCVADBackend(VADBackend):
    def __init__(self, aggressiveness: int = 1, min_speech_duration: float = 0.5, max_silence_duration: float = 1.0):
        self._vad = webrtcvad.Vad(aggressiveness)
        self.min_speech_duration = min_speech_duration
        self.max_silence_duration = max_silence_duration

    def detect(self, audio, sample_rate: int) -> list[tuple[float, float]]:
        if len(audio) == 0:
            return []

        frame_ms = 30
        frame_size = int(sample_rate * frame_ms / 1000)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)

        raw_segments: list[tuple[int, int]] = []
        in_speech = False
        silence_frames = 0
        start_index = 0

        for offset in range(0, len(pcm) - frame_size + 1, frame_size):
            frame = pcm[offset : offset + frame_size]
            is_speech = self._vad.is_speech(frame.tobytes(), sample_rate)
            if is_speech:
                silence_frames = 0
                if not in_speech:
                    in_speech = True
                    start_index = offset
            else:
                silence_frames += 1
                if in_speech and silence_frames * frame_ms / 1000 > self.max_silence_duration:
                    raw_segments.append((start_index, offset))
                    in_speech = False

        if in_speech:
            raw_segments.append((start_index, len(pcm)))

        min_samples = int(self.min_speech_duration * sample_rate)
        merged: list[tuple[int, int]] = []
        for start, end in raw_segments:
            if end - start < min_samples:
                continue
            if not merged:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if (start - prev_end) / sample_rate <= 0.5:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return [(start / sample_rate, end / sample_rate) for start, end in merged]
