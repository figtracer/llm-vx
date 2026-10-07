"""Lists the Vx kernels that will run on one GPU thread.

usage: serial_kernels.py PROGRAM.ll

PROGRAM.ll is the program compiled with `--action emit-llvm`. Vx gives a kernel
a `launch=` trip count only when it proves the kernel's loop parallel; without
one, the runtime launches a single thread. A call to anything but a few stdlib
math methods, or inline MLIR, is enough to lose it. Exits 1 if any kernel has
no trip count.
"""
import re
import sys

from kernel_times import kernel_owners

PAYLOAD = re.compile(r'@vx_npu_kernel_(\d+)_str\("vx_npu_kernel_\d+\\00([^"]{0,200})')


def main():
    path = sys.argv[1]
    owners = kernel_owners(path)
    serial = []
    count = 0
    with open(path) as f:
        for m in PAYLOAD.finditer(f.read()):
            count += 1
            if 'launch=' not in m.group(2):
                serial.append(owners.get(int(m.group(1)), '?').split('$')[0])
    print(f'{count} kernels, {len(serial)} serial' + (': ' + ', '.join(serial) if serial else ''))
    return 1 if serial else 0


if __name__ == '__main__':
    sys.exit(main())
