"""Device time per Vx kernel, from a run with VX_TIME_KERNEL=1.

usage: kernel_times.py PROGRAM.ll RUN.log [--steps N]

PROGRAM.ll is the program compiled with `--action emit-llvm`, which says which
function launches each `vx_npu_kernel_N`. RUN.log has the runtime's
"[Vx CUDA] vx_npu_kernel_N device time X ms" lines. Prints each function's
launches and total time, per step when --steps is given.
"""
import argparse
import re
from collections import defaultdict

FUNC = re.compile(r'^\s*llvm\.func @([^\s(]+)\(.*\{\s*$')
REF = re.compile(r'@vx_npu_kernel_(\d+)_str\b')
TIME = re.compile(r'\[Vx CUDA\] vx_npu_kernel_(\d+) device time ([0-9.]+) ms')


def kernel_owners(ll_path):
    """Maps each kernel number to the Vx function whose body launches it."""
    owners = {}
    current = None
    with open(ll_path) as f:
        for line in f:
            if line.startswith('  llvm.mlir.global'):
                continue
            m = FUNC.match(line)
            if m:
                current = m.group(1)
                continue
            if current:
                for k in REF.findall(line):
                    owners.setdefault(int(k), current)
    return owners


def short(name):
    """`k_add$const$...` -> `k_add`."""
    return name.split('$', 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ll')
    parser.add_argument('log')
    parser.add_argument('--steps', type=int, default=1)
    args = parser.parse_args()
    owners = kernel_owners(args.ll)
    total = defaultdict(float)
    count = defaultdict(int)
    with open(args.log, errors='replace') as f:
        for line in f:
            m = TIME.search(line)
            if m:
                name = short(owners.get(int(m.group(1)), f'vx_npu_kernel_{m.group(1)}'))
                total[name] += float(m.group(2))
                count[name] += 1
    grand = sum(total.values())
    print(f'{"function":36} {"launches":>9} {"ms/step":>10} {"share":>6}')
    for name, ms in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f'{name:36} {count[name] // args.steps:9d} {ms / args.steps:10.1f} {100 * ms / grand:5.1f}%')
    print(f'{"all Vx kernels":36} {sum(count.values()) // args.steps:9d} {grand / args.steps:10.1f}')


if __name__ == '__main__':
    main()
