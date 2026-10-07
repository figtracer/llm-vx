"""Writes an entry file that runs train_gpu.vx at one model size and batch.

usage: entry.py OUT.vx MODE B T DATA_DIR [L NH C V VP MAXT] [--aw N] [--ew N]
The model defaults to GPT-2 124M. DATA_DIR holds model.bin, train_tokens.bin,
val_tokens.bin and tokenizer.bin, as ref/make_random.c writes them. --aw and
--ew set the width of the attention matrices and of the elementwise tensors.
"""
import argparse


def shapes(B, T, L=12, NH=12, C=768, V=50257, VP=50304, MAXT=1024, aw=None, ew=2):
    """train_gpu.vx's shape parameters, in order."""
    BT = B * T
    att = L * B * NH * T * T
    # The attention matrices are 4 wide, the fastest on an L4, or wider where a
    # tensor's row count, an i32, needs it.
    if aw is None:
        aw = 4
        while att // aw >= 2**31:
            aw *= 2
    return [L, NH, C, 3 * C, 4 * C, VP, V, MAXT, B, T, BT, L * BT, att // aw, aw, L * C, L * 3 * C, L * 4 * C,
            L * BT * 4 * C // ew, BT * VP // ew, ew, VP * C // ew, MAXT * C // ew, L * 3 * C * C // ew,
            L * C * C // ew, L * 4 * C * C // ew]


def entry(mode, B, T, data, L=12, NH=12, C=768, V=50257, VP=50304, MAXT=1024, aw=None, ew=2):
    files = [f'"{data}/model.bin"', '""', '""', f'"{data}/train_tokens.bin"', f'"{data}/val_tokens.bin"',
             f'"{data}/tokenizer.bin"']
    return ('import train_gpu;\n\nfn main() -> i32 {\n  return run<' +
            ', '.join(map(str, shapes(B, T, L, NH, C, V, VP, MAXT, aw, ew))) + '>(' + str(mode) + ',\n    ' +
            ', '.join(files) + ');\n}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('mode', type=int)
    parser.add_argument('B', type=int)
    parser.add_argument('T', type=int)
    parser.add_argument('data')
    parser.add_argument('model', type=int, nargs='*')
    parser.add_argument('--aw', type=int)
    parser.add_argument('--ew', type=int, default=2)
    args = parser.parse_args()
    open(args.out, 'w').write(entry(args.mode, args.B, args.T, args.data, *args.model, aw=args.aw, ew=args.ew))
