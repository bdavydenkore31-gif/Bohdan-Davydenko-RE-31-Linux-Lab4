import argparse
import os
from pathlib import Path

import uvicorn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Водопад SDR из I/Q WAV')
    parser.add_argument('--iq', type=Path, help='Путь к WAV с сырыми I/Q')
    parser.add_argument('--meta', type=Path, help='Путь к JSON (по умолчанию рядом с WAV)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.iq:
        os.environ['SDR_IQ_FILE'] = str(args.iq.resolve())
    if args.meta:
        os.environ['SDR_META_FILE'] = str(args.meta.resolve())
    uvicorn.run('app:app', host=args.host, port=args.port, workers=1)
