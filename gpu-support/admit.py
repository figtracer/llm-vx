"""Asks Vx for the largest GPT-2 124M batch that fits each GPU, at compile time.

For each GPU and batch size this compiles train_gpu.vx against a machine file
that gives GPU_HBM a capacity, and reads the verdict from Vx's diagnostics. Two
capacities per GPU: the one Vx's fleet file declares, and what one process can
allocate on the real card, measured by vx-fit's probe.
"""
import json
import subprocess
import sys
from pathlib import Path

from entry import entry

ROOT = Path(__file__).resolve().parent.parent
VXC = ROOT / 'Vx/target/release/vxc'
BUILD = ROOT / 'build/admit'

# name: (fleet file, declared capacity, allocatable bytes measured by vx-fit)
GPUS = {
    'A100-40GB': ('a100-40', '40 GiB', 41959817216),
    'A100-80GB': ('a100-80', '80 GiB', None),
    'H100': ('h100-sxm', '80 GiB', 84460699648),
    'H200': ('h200', '141 GiB', 149554200576),
    'B200': ('b200', '192 GiB', 190800986112),
}


def machine(capacity):
    return ('Memory CPU_DRAM {}\nMemory GPU_HBM {\n  within: Memory::CPU_DRAM, capacity: ' + capacity +
            ', bandwidth: 3 TB/s, managed: cached, scope: device\n}\n')


def verdict(B, T, capacity):
    """Answers (admitted, bytes Vx counted)."""
    BUILD.mkdir(parents=True, exist_ok=True)
    src = BUILD / f'b{B}_t{T}.vx'
    src.write_text(entry(1, B, T, 'data'))
    mach = BUILD / f'cap_{capacity.replace(" ", "")}.vx'
    mach.write_text(machine(capacity))
    diag = BUILD / 'diag.json'
    diag.unlink(missing_ok=True)
    env = dict(VX_STD_PATH=f'{ROOT}/Vx/stdlib/std:{ROOT}/Vx/stdlib')
    subprocess.run(f'source Vx/config.local && {VXC} {src} --machine {mach} --action emit-mlir '
                   f'--diagnostics-json {diag} > /dev/null 2>&1', shell=True, cwd=ROOT,
                   executable='/bin/bash', env={**dict(__import__("os").environ), **env})
    d = json.loads(diag.read_text())
    sets = d.get('resident_sets') or [{}]
    counted = max(s.get('total_bytes', 0) for s in sets)
    return d['verdict'] == 'admitted', counted, d


def largest(T, capacity, hi=64):
    lo_ok, hi_bad = 0, hi + 1
    while hi_bad - lo_ok > 1:
        mid = (lo_ok + hi_bad) // 2
        ok, _, _ = verdict(mid, T, capacity)
        if ok:
            lo_ok = mid
        else:
            hi_bad = mid
    return lo_ok


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    rows = []
    for name, (fleet, declared, measured) in GPUS.items():
        b_declared = largest(T, declared)
        b_measured = largest(T, f'{measured} B') if measured else None
        rows.append((name, declared, b_declared, measured, b_measured))
        print(name, declared, b_declared, measured, b_measured, flush=True)
    out = ROOT / f'results/admission_t{T}.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([dict(gpu=r[0], declared=r[1], largest_batch_declared=r[2],
                                    allocatable_bytes=r[3], largest_batch_allocatable=r[4]) for r in rows], indent=1))


if __name__ == '__main__':
    main()
