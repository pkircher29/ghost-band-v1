"""Launch the selected native backend and Cover Studio with readiness checks."""
from pathlib import Path
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
import psutil

ROOT = Path(__file__).resolve().parents[1]
ARENAS = {'model_weight_context': 64, 'vae_weight_context': 64,
          'ar_prefill_graph_arena': 128, 'ar_decode_graph_arena': 128,
          'nar_graph_arena': 128, 'vae_graph_arena': 128}


def wait_ready(url, process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Process exited ({process.returncode}); see .runtime logs')
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.load(response)
        except (OSError, ValueError):
            time.sleep(0.3)
    raise RuntimeError(f'Timed out waiting for {url}; see .runtime logs')


def main():
    local_path = ROOT / 'launch.local.json'
    local = json.loads(local_path.read_text()) if local_path.exists() else {}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-dir', type=Path, default=local.get('engine_dir', ROOT.parent / 'audio.cpp'))
    parser.add_argument('--backend', choices=['vulkan', 'cuda', 'cpu'], default=local.get('backend', 'vulkan'))
    parser.add_argument('--device', default=local.get('device'), help='Device index or unique name fragment')
    parser.add_argument('--engine-port', type=int, default=8180)
    parser.add_argument('--app-port', type=int, default=8420)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    engine_dir = args.engine_dir.resolve()
    exe = engine_dir / f'build/windows-{args.backend}-release/bin/audiocpp_server.exe'
    if not exe.is_file():
        raise RuntimeError(f'Build the {args.backend} backend first: {exe}')
    for port in (args.engine_port, args.app_port):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', port)) == 0:
                raise RuntimeError(f'Port {port} is in use. Close the existing Cover Studio launcher first.')
    env = os.environ.copy()
    env.pop('GGML_VK_VISIBLE_DEVICES', None)
    if env.get('CUDA_PATH'):
        env['PATH'] = str(Path(env['CUDA_PATH']) / 'bin' / 'x64') + os.pathsep + env['PATH']
    listed = subprocess.run([str(exe), '--list-devices'], env=env, cwd=engine_dir,
                            capture_output=True, text=True, check=True, timeout=30)
    devices = re.findall(r'^(\w+):(\d+) "([^"]+)" \[(\w+)\]', listed.stdout, re.M)
    candidates = [d for d in devices if d[0].lower() == args.backend]
    if args.device is not None:
        selector = str(args.device)
        candidates = [d for d in candidates if d[1] == selector or selector.lower() in d[2].lower()]
        if len(candidates) != 1:
            raise RuntimeError(f'Device {selector!r} is not unique/available:\n{listed.stdout}')
    else:
        candidates.sort(key=lambda d: d[3] != 'GPU')
    if not candidates:
        raise RuntimeError(f'No {args.backend} device found:\n{listed.stdout}')
    _, device_id, device_name, _ = candidates[0]
    cfg = json.loads((engine_dir / 'server.json').read_text())
    cfg.update(host='127.0.0.1', port=args.engine_port, backend=args.backend,
               device=int(device_id), max_loaded_models=1)
    if args.backend == 'vulkan':
        env['GGML_VK_VISIBLE_DEVICES'] = device_id
        cfg['device'] = 0
    for model in cfg['models']:
        if 'path' in model:
            model['path'] = str((engine_dir / model['path']).resolve())
        if model.get('family') == 'yue2':
            options = model.setdefault('session_options', {})
            for key, value in ARENAS.items():
                options.setdefault(f'yue2.{key}_mb', str(value))
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    config_path = runtime / 'engine.json'
    config_path.write_text(json.dumps(cfg, indent=2))
    engine_url = f'http://127.0.0.1:{args.engine_port}'
    app_url = f'http://127.0.0.1:{args.app_port}'
    env.update(AUDIOCPP_URL=engine_url, ENGINE_LOG_PATH=str(runtime / 'engine.log'),
               ENGINE_DEVICE_NAME=device_name, ENGINE_BACKEND=args.backend)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    engine = app = None
    print(f'Starting {args.backend.upper()}: {device_name}', flush=True)
    try:
        with (runtime / 'engine.log').open('w') as log, (runtime / 'engine.err.log').open('w') as err:
            engine = subprocess.Popen([str(exe), '--config', str(config_path), '--log'],
                                      cwd=engine_dir, env=env, stdout=log, stderr=err, creationflags=flags)
        ready = wait_ready(engine_url + '/health', engine)
        if ready.get('backend') != args.backend:
            raise RuntimeError(f'Unexpected engine backend: {ready}')
        env['ENGINE_PID'] = str(engine.pid)
        with (runtime / 'app.log').open('w') as log, (runtime / 'app.err.log').open('w') as err:
            app = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'server:app', '--host', '0.0.0.0',
                                    '--port', str(args.app_port)], cwd=ROOT / 'app', env=env,
                                   stdout=log, stderr=err, creationflags=flags)
        health = wait_ready(app_url + '/api/health', app)
        if not health.get('ok'):
            raise RuntimeError('Cover Studio cannot reach the engine')
        (runtime / 'processes.json').write_text(json.dumps({'launcher': os.getpid(), 'engine': engine.pid, 'app': app.pid}))
        print(f'Cover Studio ready: {app_url} ({device_name}). Ctrl+C to stop.', flush=True)
        if not args.no_browser:
            webbrowser.open(app_url)
        while app.poll() is None:
            if engine.poll() is not None:
                time.sleep(1)
                continue
            time.sleep(0.5)
    finally:
        for process in (app, engine):
            if process and process.poll() is None:
                # Windows venv executables can spawn a Python redirector child.
                # Stop that owned tree as well, or its server port stays bound.
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in reversed(children):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for child in children:
                    try:
                        child.wait(timeout=5)
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.TimeoutExpired:
                        child.kill()
                    except psutil.AccessDenied:
                        # Windows can deny OpenProcess after the child exits.
                        if child.is_running():
                            print(f'Could not verify child {child.pid} stopped', file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Cover Studio stopped.')
    except Exception as exc:
        print(f'Cannot start Cover Studio: {exc}', file=sys.stderr)
        sys.exit(1)
