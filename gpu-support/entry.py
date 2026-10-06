"""Writes an entry file that runs train_gpu.vx at one model size and batch.

usage: entry.py OUT.vx MODE B T DATA_DIR [L NH C V VP MAXT]
The model defaults to GPT-2 124M. DATA_DIR holds model.bin, train_tokens.bin,
val_tokens.bin and tokenizer.bin, as ref/make_random.c writes them.
"""
import sys


def entry(mode, B, T, data, L=12, NH=12, C=768, V=50257, VP=50304, MAXT=1024):
    BT = B * T
    shapes = [L, NH, C, 3 * C, 4 * C, VP, V, MAXT, B, T, BT, L * BT, L * B * NH * T, L * C, L * 3 * C, L * 4 * C,
              L * BT * 4 * C // 32, BT * VP // 32]
    files = [f'"{data}/model.bin"', '""', '""', f'"{data}/train_tokens.bin"', f'"{data}/val_tokens.bin"',
             f'"{data}/tokenizer.bin"']
    return ('import train_gpu;\n\nfn main() -> i32 {\n  return run<' + ', '.join(map(str, shapes)) + '>(' +
            str(mode) + ',\n    ' + ', '.join(files) + ');\n}\n')


if __name__ == '__main__':
    out, mode, B, T, data = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    model = [int(x) for x in sys.argv[6:]]
    open(out, 'w').write(entry(mode, B, T, data, *model))
