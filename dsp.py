"""Bounded-memory spectrum extraction from stereo unsigned-8-bit IQ WAV."""
from functools import lru_cache
import json
import math
from pathlib import Path
import threading
import wave

import numpy as np

FFT_SIZES = (512, 1024, 2048, 4096, 8192)
DB_MIN, DB_MAX = -120.0, 0.0


class IQRecording:
    def __init__(self, path: Path, metadata_path: Path | None = None):
        self.path = Path(path)
        metadata_path = Path(metadata_path) if metadata_path else self.path.with_suffix('.json')
        with metadata_path.open(encoding='utf-8-sig') as f:
            metadata = json.load(f)
        with wave.open(str(self.path), 'rb') as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (2, 1, 'NONE'):
                raise ValueError('Нужен PCM WAV: 2 канала I/Q, unsigned 8-bit. Аудио mono не подходит.')
            self.sample_rate = wav.getframerate()
            self.samples = wav.getnframes()
            if self.samples:
                wav.setpos(self.samples - 1)
                if len(wav.readframes(1)) != 2:
                    raise ValueError('WAV обрезан: последние I/Q-отсчёты отсутствуют.')
        if self.samples == 0:
            raise ValueError('WAV не содержит I/Q-отсчётов.')
        if int(metadata.get('sample_rate', self.sample_rate)) != self.sample_rate:
            raise ValueError('Частота дискретизации в JSON отличается от WAV.')
        if int(metadata.get('bits', 8)) != 8:
            raise ValueError('JSON должен описывать 8-битные I/Q-данные.')
        self.center = float(metadata['center_frequency_hz'])
        if not math.isfinite(self.center) or self.center <= 0:
            raise ValueError('Некорректная центральная частота в JSON.')
        self.duration = self.samples / self.sample_rate
        self.device = str(metadata.get('device', 'SDR'))
        self.started = str(metadata.get('started', ''))
        self._limit = threading.BoundedSemaphore(2)
        self.chunk = lru_cache(maxsize=8)(self._chunk)

    def info(self):
        return {
            'file': self.path.name, 'sample_rate': self.sample_rate,
            'center_frequency_hz': self.center, 'samples': self.samples,
            'duration_seconds': self.duration, 'device': self.device,
            'started': self.started, 'format': 'CU8 stereo WAV, I + jQ',
            'frequency_min_hz': self.center - self.sample_rate / 2,
            'frequency_max_hz': self.center + self.sample_rate / 2,
            'fft_sizes': FFT_SIZES, 'db_min': DB_MIN, 'db_max': DB_MAX,
            'averages': 4,
        }

    def total_rows(self, rows_per_second: int):
        return math.ceil(self.samples * rows_per_second / self.sample_rate)

    def _chunk(self, fft_size: int, rows_per_second: int, start: int, count: int, remove_dc: bool):
        if fft_size not in FFT_SIZES or not 5 <= rows_per_second <= 60:
            raise ValueError('Недопустимые FFT или число строк в секунду.')
        if start < 0 or not 1 <= count <= 128:
            raise ValueError('Недопустимый диапазон строк.')
        count = max(0, min(count, self.total_rows(rows_per_second) - start))
        result = np.empty((count, fft_size), dtype=np.uint8)
        window = np.hanning(fft_size).astype(np.float32)
        normalization = float(window.sum()) ** 2
        with self._limit, wave.open(str(self.path), 'rb') as wav:
            for index in range(count):
                row = start + index
                begin = row * self.sample_rate // rows_per_second
                end = min((row + 1) * self.sample_rate // rows_per_second, self.samples)
                # Four windows spread across a time bin. Average power before log.
                last = max(begin, end - fft_size)
                positions = np.linspace(begin, last, 4, dtype=np.int64)
                frames = np.zeros((4, fft_size), dtype=np.complex64)
                for k, position in enumerate(positions):
                    wav.setpos(int(position))
                    raw = wav.readframes(min(fft_size, end - int(position)))
                    iq = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5
                    frames[k, :len(iq) // 2] = iq[0::2] + 1j * iq[1::2]
                    if remove_dc and len(iq):
                        length = len(iq) // 2
                        frames[k, :length] -= frames[k, :length].mean()
                spectrum = np.fft.fft(frames * window, axis=1)
                power = np.mean(np.abs(spectrum) ** 2, axis=0) / normalization
                db = 10 * np.log10(np.maximum(np.fft.fftshift(power), 1e-12))
                result[index] = np.rint(np.clip((db - DB_MIN) * 255 / (DB_MAX - DB_MIN), 0, 255))
        return result.tobytes(), count
