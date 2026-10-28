"""FastAPI server. Run with python run.py --iq /path/to/baseband.wav."""
from contextlib import asynccontextmanager
import os
import wave
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from dsp import FFT_SIZES, IQRecording
from analysis2d import Analyzer, AnalysisUnavailable, tensorflow_status

ROOT = Path(__file__).resolve().parent
DEFAULT_IQ = ROOT / 'data' / 'baseband_103600000Hz_19-36-09_27-10-2026.wav'


def create_app(iq_path=None, metadata_path=None):
    @asynccontextmanager
    async def lifespan(application):
        source = Path(iq_path or os.environ.get('SDR_IQ_FILE', str(DEFAULT_IQ)))
        metadata = metadata_path or os.environ.get('SDR_META_FILE')
        try:
            application.state.recording = IQRecording(source, metadata)
        except (OSError, ValueError, KeyError, EOFError, wave.Error) as error:
            raise RuntimeError(f'Не удалось открыть запись {source}: {error}') from error
        application.state.analyzer = Analyzer(application.state.recording)
        yield
        application.state.analyzer.run.cache_clear()
        application.state.recording.chunk.cache_clear()

    application = FastAPI(title='SDR Waterfall', lifespan=lifespan)

    @application.get('/', include_in_schema=False)
    def index():
        return FileResponse(ROOT / 'static' / 'index.html')

    @application.get('/api/info')
    def info():
        return application.state.recording.info()

    @application.get('/api/waterfall')
    def waterfall(
        fft_size: int = 2048,
        rows_per_second: int = Query(40, ge=5, le=60),
        start: int = Query(0, ge=0),
        count: int = Query(128, ge=1, le=128),
        remove_dc: bool = False,
    ):
        if fft_size not in FFT_SIZES:
            raise HTTPException(422, f'FFT должен быть одним из {FFT_SIZES}')
        recording = application.state.recording
        payload, returned = recording.chunk(fft_size, rows_per_second, start, count, remove_dc)
        return Response(payload, media_type='application/octet-stream', headers={
            'X-Rows': str(returned), 'X-Bins': str(fft_size),
            'X-Start-Row': str(start), 'X-Total-Rows': str(recording.total_rows(rows_per_second)),
            'X-DB-Min': '-120', 'X-DB-Max': '0', 'Cache-Control': 'no-store',
        })

    @application.get('/api/analysis/status')
    def analysis_status():
        return tensorflow_status()

    @application.get('/api/analysis')
    def analysis(
        start: int = Query(0, ge=0),
        count: int = Query(200, ge=1, le=256),
        threshold_db: float = Query(10.0, ge=3, le=30),
        min_area: int = Query(12, ge=1, le=1000),
        remove_dc: bool = False,
    ):
        try:
            return application.state.analyzer.run(start, count, threshold_db, min_area, remove_dc)
        except AnalysisUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return application


app = create_app()
