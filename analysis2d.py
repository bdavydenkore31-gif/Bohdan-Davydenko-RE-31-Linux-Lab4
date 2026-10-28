"""TensorFlow time-frequency filtering; no trained classifier is used."""
import base64
from functools import lru_cache
import importlib.util
import os
import threading

import numpy as np

ANALYSIS_FFT = 2048
ANALYSIS_FPS = 20
ANALYSIS_BINS = 512
_TF_LOCK = threading.Lock()
_TF = None


class AnalysisUnavailable(RuntimeError):
    pass


def tensorflow_status():
    installed = importlib.util.find_spec('tensorflow') is not None
    return {'installed': installed, 'loaded': _TF is not None,
            'version': getattr(_TF, '__version__', None),
            'method': '2D convolution + threshold + connected regions',
            'trained_model': False, 'fft_size': ANALYSIS_FFT,
            'rows_per_second': ANALYSIS_FPS, 'bins': ANALYSIS_BINS,
            'install_command': 'python -m pip install -r requirements-tensorflow.txt'}


def load_tensorflow():
    global _TF
    with _TF_LOCK:
        if _TF is None:
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
            os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
            try:
                import tensorflow as tf
                _TF = tf
            except (ImportError, OSError) as error:
                raise AnalysisUnavailable(
                    'TensorFlow не установлен или не загружается. Выполните в активной .venv: '
                    'python -m pip install -r requirements-tensorflow.txt. '
                    f'Причина: {error}'
                ) from error
    return _TF


def filter_2d(db, threshold_db):
    """Return smoothed excess above row median and threshold mask, using real TF ops."""
    tf = load_tensorflow()
    tensor = tf.convert_to_tensor(db, dtype=tf.float32)[None, :, :, None]
    # Reduce 2048 frequency bins to 512 by averaging LINEAR power, not dB.
    power = tf.pow(10.0, tensor / 10.0)
    power = tf.nn.avg_pool2d(power, ksize=[1, 4], strides=[1, 4], padding='VALID')
    reduced_db = 10.0 * tf.math.log(tf.maximum(power, 1e-12)) / tf.math.log(10.0)
    sorted_db = tf.sort(reduced_db, axis=2)
    noise = (sorted_db[:, :, 255:256, :] + sorted_db[:, :, 256:257, :]) / 2
    excess = reduced_db - noise
    # A fixed separable 3 x 5 Gaussian-like kernel acts on BOTH time and frequency.
    kernel = np.outer([1., 2., 1.], [1., 4., 6., 4., 1.]).astype(np.float32)
    kernel /= kernel.sum()
    padded = tf.pad(excess, [[0, 0], [1, 1], [2, 2], [0, 0]], mode='SYMMETRIC')
    smooth = tf.nn.conv2d(padded, kernel[:, :, None, None], strides=1, padding='VALID')
    score = smooth[0, :, :, 0].numpy()
    mask = tf.greater_equal(smooth, float(threshold_db))[0, :, :, 0].numpy()
    return score, mask, noise[0, :, 0, 0].numpy()


def connected_regions(mask, score, min_area):
    """Four-neighbor components; suppress components below min_area cells."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    kept = np.zeros_like(mask, dtype=np.uint8)
    regions = []
    for y, x in zip(*np.nonzero(mask)):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        visited[y, x] = True
        stack, points = [(y, x)], []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(points) < min_area:
            continue
        yy, xx = np.array(points).T
        kept[yy, xx] = 255
        regions.append({'x0': int(xx.min()), 'x1': int(xx.max())+1,
                        'y0': int(yy.min()), 'y1': int(yy.max())+1,
                        'cells': len(points), 'peak_excess_db': round(float(score[yy, xx].max()), 2)})
    regions.sort(key=lambda r: r['cells'], reverse=True)
    return kept, regions


class Analyzer:
    def __init__(self, recording):
        self.recording = recording
        self._lock = threading.Lock()
        self.run = lru_cache(maxsize=4)(self._run)

    def _run(self, start, count, threshold_db, min_area, remove_dc):
        rec = self.recording
        if start < 0 or start >= rec.total_rows(ANALYSIS_FPS):
            raise ValueError('Начало анализа должно быть внутри записи.')
        if not 1 <= count <= 256 or not 3 <= threshold_db <= 30 or not 1 <= min_area <= 1000:
            raise ValueError('Недопустимые параметры анализа.')
        with self._lock:
            tf = load_tensorflow()
            rows = []
            for offset in range(0, count, 128):
                payload, n = rec.chunk(ANALYSIS_FFT, ANALYSIS_FPS, start+offset, min(128,count-offset), remove_dc)
                if not n:
                    break
                rows.append(np.frombuffer(payload,dtype=np.uint8).reshape(n,ANALYSIS_FFT))
            db = -120.0 + np.concatenate(rows).astype(np.float32) * (120.0/255.0)
            score, mask, noise = filter_2d(db, threshold_db)
            kept, regions = connected_regions(mask, score, min_area)
            dt, df = 1/ANALYSIS_FPS, rec.sample_rate/ANALYSIS_BINS
            left = rec.center-rec.sample_rate/2
            for region in regions:
                region.update({
                    'frequency_min_hz': left+region['x0']*df,
                    'frequency_max_hz': left+region['x1']*df,
                    'bandwidth_hz': (region['x1']-region['x0'])*df,
                    'time_start_s': (start+region['y0'])*dt,
                    'time_end_s': min(rec.duration,(start+region['y1'])*dt),
                })
            score_bytes = np.rint(np.clip(score,0,30)*(255/30)).astype(np.uint8).tobytes()
            return {
                'engine': 'TensorFlow', 'version': tf.__version__, 'trained_model': False,
                'method': 'Fixed 3x5 convolution of excess above row median; threshold; 4-neighbor regions',
                'rows': int(db.shape[0]), 'bins': ANALYSIS_BINS,
                'fft_size': ANALYSIS_FFT, 'rows_per_second': ANALYSIS_FPS,
                'start_row': start, 'time_start_s': start*dt,
                'time_end_s': min(rec.duration,(start+db.shape[0])*dt),
                'frequency_min_hz': left, 'frequency_max_hz': left+rec.sample_rate,
                'frequency_step_hz': df, 'threshold_db': threshold_db, 'min_area': min_area,
                'remove_dc': remove_dc, 'noise_median_dbfs': round(float(np.median(noise)),2),
                'occupied_percent': round(float(np.count_nonzero(kept)/kept.size*100),2),
                'region_count': len(regions), 'regions': regions[:64],
                'regions_omitted': max(0,len(regions)-64),
                'score_min_db': 0, 'score_max_db': 30,
                'score_u8_b64': base64.b64encode(score_bytes).decode('ascii'),
                'mask_u8_b64': base64.b64encode(kept.tobytes()).decode('ascii'),
            }
