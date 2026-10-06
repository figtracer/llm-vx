"""Runs a job on a rented GPU through Fission: build Vx with CUDA, run the job, collect /workspace/out.

usage: rent.py GPU JOB [--duration 3h] [--work 2h] [--budget 4] [--keep] [--approve]

Without --approve it prints the quote and stops. A failed or unclear purchase
is never retried: Fission keeps the record, and the next attempt needs a new
name and a new decision. With --keep, a failed job leaves the prepaid sandbox
open so a fix can run there; Fission uploads never overwrite, so upload fixed
files under new names, and close the sandbox when done.
"""
import argparse
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/runs'
SKIP = {'.git', 'Vx', 'llm.c', 'build', 'data', 'results'}
SUBMODULES = {'Vx': 'https://github.com/vx-lang/Vx', 'llm.c': 'https://github.com/karpathy/llm.c'}


def fission(*args, check=True):
    result = subprocess.run(['fission', 'advanced', *args], capture_output=True, text=True)
    print('$ fission advanced', ' '.join(args[:3]), '->', result.returncode, flush=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    text = result.stdout.strip()
    return json.loads(text) if text.startswith('{') else text


def package(path):
    """The repository without its submodules, plus the submodule commits for the sandbox to fetch."""
    def keep(info):
        parts = Path(info.name).parts
        return None if any(p in SKIP for p in parts[1:2]) else info
    pins = ''.join(f"{d} {url} {subprocess.run(['git', '-C', str(ROOT / d), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()}\n"
                   for d, url in SUBMODULES.items())
    (ROOT / 'build').mkdir(exist_ok=True)
    (ROOT / 'build/submodules.txt').write_text(pins)
    with tarfile.open(path, 'w:gz') as tar:
        tar.add(ROOT, arcname='llm-vx', filter=keep)
        tar.add(ROOT / 'build/submodules.txt', arcname='llm-vx/submodules.txt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu')
    parser.add_argument('job')
    parser.add_argument('--duration', default='3h')
    parser.add_argument('--work', default='2h')
    parser.add_argument('--budget', default='4')
    parser.add_argument('--name')
    parser.add_argument('--keep', action='store_true')
    parser.add_argument('--approve', action='store_true')
    args = parser.parse_args()
    job = ROOT / 'gpu-support/jobs' / f'{args.job}.sh'
    name = args.name or f'llmvx-{args.gpu.lower()}-{args.job}-{time.strftime("%m%d%H%M")}'

    plan = fission('plan', name, '--provider', 'modal-tempo', '--gpu', args.gpu, '--duration', args.duration,
                   '--budget', args.budget)
    print(json.dumps({k: plan.get(k) for k in ['id', 'creationQuote', 'totalCap', 'durationSeconds']}))
    if not args.approve:
        return print('Preview only; repeat with --approve to pay.')

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / 'llm-vx.tgz'
    package(bundle)
    status = None
    failed = False
    try:
        state = fission('open', name, '--plan', plan['id'], '--approve')
        if not (state.get('guestGpu') or {}).get('verified'):
            raise RuntimeError(f"GPU check failed: {state.get('preparationError')}")
        print(json.dumps(state['guestGpu']['devices']))
        fission('wait', name, 'bootstrap', '--duration', '10m', '--max-spend', '0.01')
        fission('upload', name, str(bundle), '/workspace/llm-vx.tgz')
        fission('upload', name, str(ROOT / 'gpu-support/remote.sh'), '/workspace/remote.sh')
        fission('run', name, args.job, '--duration', args.work, '--',
                'bash', '/workspace/remote.sh', 'bash', f'gpu-support/jobs/{args.job}.sh')
        result = fission('wait', name, args.job, '--duration', args.work, '--max-spend', '0.05', check=False)
        print(json.dumps(result if isinstance(result, str) else {k: result.get(k) for k in ['phase', 'waitingStopped']}))
        failed = not isinstance(result, dict) or result.get('phase') != 'succeeded'
        fission('download', name, f'/workspace/.fission/jobs/{args.job}/output.log', str(out / 'output.log'),
                check=False)
        # Downloads come back 48 KB per paid call, so only small text results.
        for remote in ['test_gpu_small.log', 'vx_gpu_train.txt', 'compare_small.txt', 'probe.txt',
                       'bench_b4.log', 'bench_b5.log', 'results.txt']:
            fission('download', name, f'/workspace/out/{remote}', str(out / remote), check=False)
    except Exception:
        failed = True
        raise
    finally:
        if failed and args.keep:
            print(f'kept {name} open; close it with: fission advanced close {name} --discard-output')
            return
        fission('close', name, '--discard-output', check=False)
        for _ in range(8):
            status = fission('status', name, '--refresh', '--json', check=False)
            if isinstance(status, dict) and status.get('phase') == 'terminated':
                break
            time.sleep(20)
        print('cleanup:', status.get('phase') if isinstance(status, dict) else status)


if __name__ == '__main__':
    sys.exit(main())
