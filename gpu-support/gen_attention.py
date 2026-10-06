"""Generates attention.vx: the attention kernels whose inner loops keep K values
per thread in registers, unrolled by hand because Vx has no local arrays.

Each kernel has two bodies: K = 64, one pass for GPT-2's head size of 64, and
K = 16, as many passes as hs / 16 for any head size that is a multiple of 16.
Both sum in the same order as llm.c, so they give the same bits.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lanes(k, fmt, indent):
    return '\n'.join(indent + fmt.format(j=j, jj=(f' + {j}' if j else '')) for j in range(k))


def body_switch(make):
    """`if hs == 64 { 64 lanes } else { 16 lanes }`, at the kernel's indentation."""
    return ('        if hs == 64 {\n' + make(64, '          ') + '\n        } else {\n' + make(16, '          ') +
            '\n        }')


def scores_dot(k, ind):
    """preatt[r][t2] = scale * (q . key[t2]) for t2 <= t, and maxval."""
    if k == 64:
        return (lanes(k, 'let q{j} = qkv[qrow][c0{jj}];', ind) + f'''
{ind}for t2 in 0..(t + 1) {{
{ind}  let kr = row0 + t2;
{ind}  let mut val : f32 = 0.0;
''' + lanes(k, 'val = val + q{j} * qkv[kr][C + c0{jj}];', ind + '  ') + f'''
{ind}  val = val * scale;
{ind}  if val > maxval {{ maxval = val; }}
{ind}  preatt[r][t2] = val;
{ind}}}''')
    return (f'''{ind}for chunk in 0..(hs / 16) {{
{ind}  let c = c0 + chunk * 16;
''' + lanes(k, 'let q{j} = qkv[qrow][c{jj}];', ind + '  ') + f'''
{ind}  for t2 in 0..(t + 1) {{
{ind}    let kr = row0 + t2;
{ind}    let mut val : f32 = 0.0;
{ind}    if chunk > 0 {{ val = preatt[r][t2]; }}
''' + lanes(k, 'val = val + q{j} * qkv[kr][C + c{jj}];', ind + '    ') + f'''
{ind}    preatt[r][t2] = val;
{ind}  }}
{ind}}}
{ind}for t2 in 0..(t + 1) {{
{ind}  let val = preatt[r][t2] * scale;
{ind}  if val > maxval {{ maxval = val; }}
{ind}  preatt[r][t2] = val;
{ind}}}''')


SCORES = '''// Attention scores for one (batch, head, position): q . k for each key at or
// before it, scaled, then the softmax over them. One thread per row of att.
// The query's components stay in registers, so the row of keys is the only
// thing read per key. Nothing reads att above the diagonal, so it is not
// written. hs must be 64 or a multiple of 16.
fn k_attention_scores<const RA : i32, const T : i32, const R : i32, const C3 : i32>(
    preatt : &mut Tensor<f32, [RA, T], Memory::GPU_HBM>, att : &mut Tensor<f32, [RA, T], Memory::GPU_HBM>,
    qkv : &Tensor<f32, [R, C3], Memory::GPU_HBM>, NH : i32, BT : i32, l : i32, scale : f32,
    lo : i32, n : i32) -> void {
  let C = C3 / 3;
  let hs = C / NH;
  spawn on(Topology::GPU) {
    for r in 0..RA {
      if r >= lo && r < lo + n {
        let rr = r - lo;
        let b = rr / (NH * T);
        let h = (rr / T) % NH;
        let t = rr % T;
        let qrow = l * BT + b * T + t;
        let row0 = l * BT + b * T;
        let c0 = h * hs;
        let mut maxval : f32 = -10000.0;
@DOT@
        let mut expsum : f32 = 0.0;
        for t2 in 0..(t + 1) {
          let expv = (preatt[r][t2] - maxval).exp();
          expsum = expsum + expv;
          att[r][t2] = expv;
        }
        let mut expsum_inv : f32 = 0.0;
        if expsum != 0.0 { expsum_inv = 1.0 / expsum; }
        for t2 in 0..(t + 1) {
          att[r][t2] = att[r][t2] * expsum_inv;
        }
      }
    }
  }
  return;
}
'''


def out_body(k, ind):
    """scratch[r][i] = sum over t2 <= t of att[t2] * value[t2][i]."""
    inner = (f'''{ind}for t2 in 0..(t + 1) {{
{ind}  let w = att[r][t2];
{ind}  let v = row0 + t2;
''' + lanes(k, 's{j} = s{j} + w * qkv[v][c{jj}];', ind + '  ') + f'''
{ind}}}
''' + lanes(k, 'scratch[r][i0{jj}] = s{j};', ind))
    if k == 64:
        return (f'{ind}let i0 = 0;\n{ind}let c = C * 2 + c0;\n' + lanes(k, 'let mut s{j} : f32 = 0.0;', ind) + '\n' +
                inner)
    return (f'''{ind}for chunk in 0..(hs / 16) {{
{ind}  let i0 = chunk * 16;
{ind}  let c = C * 2 + c0 + i0;
''' + lanes(k, 'let mut s{j} : f32 = 0.0;', ind + '  ') + '\n' +
            '\n'.join('  ' + ln for ln in inner.split('\n')) + f'\n{ind}}}')


OUT = '''// The attention output of one (batch, head, position): out[i] = the sum over
// t2 <= t of att[t2] * value[t2][i], into the first hs columns of that row of
// `scratch`. The caller passes preatt, which nothing reads once att is computed.
// One thread per row of att, K outputs per pass over the row.
fn k_attention_out_head<const RA : i32, const T : i32, const R : i32, const C3 : i32>(
    scratch : &mut Tensor<f32, [RA, T], Memory::GPU_HBM>, att : &Tensor<f32, [RA, T], Memory::GPU_HBM>,
    qkv : &Tensor<f32, [R, C3], Memory::GPU_HBM>, NH : i32, BT : i32, l : i32, lo : i32, n : i32) -> void {
  let C = C3 / 3;
  let hs = C / NH;
  spawn on(Topology::GPU) {
    for r in 0..RA {
      if r >= lo && r < lo + n {
        let rr = r - lo;
        let b = rr / (NH * T);
        let h = (rr / T) % NH;
        let t = rr % T;
        let row0 = l * BT + b * T;
        let c0 = h * hs;
@BODY@
      }
    }
  }
  return;
}
'''


def bscores_body(k, ind):
    """datt[r][t2] = value[t2] . dout for t2 <= t, and dot += att * datt."""
    if k == 64:
        return (lanes(k, 'let d{j} = dout[o][c0{jj}];', ind) + f'''
{ind}for t2 in 0..(t + 1) {{
{ind}  let v = row0 + t2;
{ind}  let mut acc : f32 = 0.0;
''' + lanes(k, 'acc = acc + qkv[v][C * 2 + c0{jj}] * d{j};', ind + '  ') + f'''
{ind}  datt[r][t2] = acc;
{ind}  dot = dot + att[r][t2] * acc;
{ind}}}''')
    return (f'''{ind}for chunk in 0..(hs / 16) {{
{ind}  let c = c0 + chunk * 16;
''' + lanes(k, 'let d{j} = dout[o][c{jj}];', ind + '  ') + f'''
{ind}  for t2 in 0..(t + 1) {{
{ind}    let v = row0 + t2;
{ind}    let mut acc : f32 = 0.0;
{ind}    if chunk > 0 {{ acc = datt[r][t2]; }}
''' + lanes(k, 'acc = acc + qkv[v][C * 2 + c{jj}] * d{j};', ind + '    ') + f'''
{ind}    datt[r][t2] = acc;
{ind}  }}
{ind}}}
{ind}for t2 in 0..(t + 1) {{ dot = dot + att[r][t2] * datt[r][t2]; }}''')


BSCORES = '''// For one query position: datt from the values, then dpreatt through the
// softmax in O(T), as llm.c's CUDA version computes it:
// dpreatt[t3] = att[t3] * (datt[t3] - sum over t2 of att[t2] * datt[t2]).
// llm.c's CPU code sums the O(T^2) form, which differs only in rounding and
// costs about T^3 / 3 multiply-adds per head. Both are assigned, not
// accumulated, so neither needs zeroing before a step; nothing reads them above
// the diagonal. dout's components stay in registers, as the query's do in the
// forward scores.
fn k_attention_backward_scores<const RA : i32, const T : i32, const R : i32, const C3 : i32, const C : i32>(
    dpreatt : &mut Tensor<f32, [RA, T], Memory::GPU_HBM>, datt : &mut Tensor<f32, [RA, T], Memory::GPU_HBM>,
    att : &Tensor<f32, [RA, T], Memory::GPU_HBM>, qkv : &Tensor<f32, [R, C3], Memory::GPU_HBM>,
    dout : &Tensor<f32, [R, C], Memory::GPU_HBM>, NH : i32, BT : i32, l : i32, lo : i32, n : i32) -> void {
  let hs = C / NH;
  spawn on(Topology::GPU) {
    for r in 0..RA {
      if r >= lo && r < lo + n {
        let rr = r - lo;
        let b = rr / (NH * T);
        let h = (rr / T) % NH;
        let t = rr % T;
        let o = l * BT + b * T + t;
        let row0 = l * BT + b * T;
        let c0 = h * hs;
        let mut dot : f32 = 0.0;
@BODY@
        for t3 in 0..(t + 1) {
          dpreatt[r][t3] = att[r][t3] * (datt[r][t3] - dot);
        }
      }
    }
  }
  return;
}
'''


def part_body(k, ind):
    """One part of dqkv into scratch[r][0..hs), K columns per pass."""
    def sums(src, ind2):
        if src == 'q':
            return f'''{ind2}for t2 in 0..(t + 1) {{
{ind2}  let d = dpreatt[r][t2];
{ind2}  let kr = row0 + t2;
''' + lanes(k, 's{j} = s{j} + qkv[kr][c{jj}] * d * scale;', ind2 + '  ') + f'\n{ind2}}}'
        if src == 'k':
            return f'''{ind2}for tq in 0..T {{
{ind2}  if tq >= t {{
{ind2}    let d = dpreatt[arow + tq][t];
{ind2}    let q = row0 + tq;
''' + lanes(k, 's{j} = s{j} + qkv[q][c{jj}] * d * scale;', ind2 + '    ') + f'\n{ind2}  }}\n{ind2}}}'
        return f'''{ind2}for tq in 0..T {{
{ind2}  if tq >= t {{
{ind2}    let w = att[arow + tq][t];
{ind2}    let q = row0 + tq;
''' + lanes(k, 's{j} = s{j} + w * dout[q][c{jj}];', ind2 + '    ') + f'\n{ind2}  }}\n{ind2}}}'

    def one(ind2, i0):
        return (f'{ind2}let i0 = {i0};\n' + lanes(k, 'let mut s{j} : f32 = 0.0;', ind2) + f'''
{ind2}if part == 0 {{
{ind2}  let c = C + c0 + i0;
''' + sums('q', ind2 + '  ') + f'''
{ind2}}} else if part == 1 {{
{ind2}  let c = c0 + i0;
''' + sums('k', ind2 + '  ') + f'''
{ind2}}} else {{
{ind2}  let c = c0 + i0;
''' + sums('v', ind2 + '  ') + f'''
{ind2}}}
''' + lanes(k, 'scratch[r][i0{jj}] = s{j};', ind2))

    if k == 64:
        return one(ind, '0')
    return f'{ind}for chunk in 0..(hs / 16) {{\n' + one(ind + '  ', 'chunk * 16') + f'\n{ind}}}'


PART = '''// One part of dqkv (0 dquery, 1 dkey, 2 dvalue) for one (batch, head, position),
// into the first hs columns of that row of `scratch`. The caller passes datt,
// which no later kernel reads. One thread per row of att: twelve times as many
// threads as one per position. dquery sums over the keys before the position,
// dkey and dvalue over the queries at or after it. The second loop runs over
// every query and skips the earlier ones, so a warp reads one row of att and
// dpreatt at a time. K outputs per pass.
fn k_attention_backward_part<const RA : i32, const T : i32, const R : i32, const C3 : i32, const C : i32>(
    scratch : &mut Tensor<f32, [RA, T], Memory::GPU_HBM>, dpreatt : &Tensor<f32, [RA, T], Memory::GPU_HBM>,
    att : &Tensor<f32, [RA, T], Memory::GPU_HBM>, qkv : &Tensor<f32, [R, C3], Memory::GPU_HBM>,
    dout : &Tensor<f32, [R, C], Memory::GPU_HBM>, NH : i32, BT : i32, l : i32, scale : f32, part : i32,
    lo : i32, n : i32) -> void {
  let hs = C / NH;
  spawn on(Topology::GPU) {
    for r in 0..RA {
      if r >= lo && r < lo + n {
        let rr = r - lo;
        let b = rr / (NH * T);
        let h = (rr / T) % NH;
        let t = rr % T;
        let arow = lo + b * NH * T + h * T;
        let row0 = l * BT + b * T;
        let c0 = h * hs;
@BODY@
      }
    }
  }
  return;
}
'''

HEADER = '''// Generated by gpu-support/gen_attention.py; edit that instead.
//
// The attention kernels of the GPU program, with the values each thread reuses
// across a row of att held in registers: 64 of them for a head size of 64, else
// 16 per pass for any head size that is a multiple of 16. Vx has no local
// arrays, so the 64 are 64 named variables. Every sum runs in llm.c's order.
'''


def main():
    src = (HEADER + '\n' + SCORES.replace('@DOT@', body_switch(scores_dot)) + '\n' +
           OUT.replace('@BODY@', body_switch(out_body)) + '\n' +
           BSCORES.replace('@BODY@', body_switch(bscores_body)) + '\n' +
           PART.replace('@BODY@', body_switch(part_body)))
    open(os.path.join(ROOT, 'attention.vx'), 'w').write(src)


if __name__ == '__main__':
    main()
