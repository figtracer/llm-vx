"""Compares the losses and sampled text of llm.c's train_gpt2 and the Vx port.

llm.c prints losses with "%f". Vx prints the shortest text that reads back as
the same f32, so each Vx loss is read back as an f32 and printed with "%f".
"""
import re
import struct
import sys


def parse(path):
    text = open(path, 'rb').read().decode('utf-8', 'replace')
    val = re.findall(r'^val loss (\S+)$', text, re.M)
    train = re.findall(r'^step (\d+): train loss (\S+) ', text, re.M)
    samples = re.findall(r'generating:\n---\n(.*?)\n---\n', text, re.S)
    return val, [loss for _, loss in train], samples


def as_c(loss):
    f32 = struct.unpack('<f', struct.pack('<f', float(loss)))[0]
    return '%f' % f32


c_val, c_train, c_samples = parse(sys.argv[1])
vx_val, vx_train, vx_samples = parse(sys.argv[2])
ok = True
for name, c, vx in [('val loss', c_val, vx_val), ('train loss', c_train, vx_train)]:
    vx = [as_c(x) for x in vx]
    same = c == vx and len(c) > 0
    ok = ok and same
    print(f'{name}: {len(c)} values, {"identical" if same else "DIFFERENT"}')
    if not same:
        for i, (a, b) in enumerate(zip(c, vx)):
            if a != b:
                print(f'  first difference at {i}: llm.c {a}, vx {b}')
                break
same = c_samples == vx_samples and len(c_samples) > 0
ok = ok and same
print(f'samples: {len(c_samples)} blocks, {"identical" if same else "DIFFERENT"}')
print('overall okay:', int(ok))
sys.exit(0 if ok else 1)
