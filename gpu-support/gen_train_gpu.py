# Generates train_gpu.vx from train_gpu.tmpl: the template holds the logic, and
# the tables below expand into the per-tensor declarations, zeroing, AdamW and
# gradient checks.
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params = [  # name, rows, cols
    ('wte', 'VP', 'C'), ('wpe', 'MAXT', 'C'), ('ln1w', 'LC', '1'), ('ln1b', 'LC', '1'),
    ('qkvw', 'LC3', 'C'), ('qkvb', 'LC3', '1'), ('attprojw', 'LC', 'C'), ('attprojb', 'LC', '1'),
    ('ln2w', 'LC', '1'), ('ln2b', 'LC', '1'), ('fcw', 'LC4', 'C'), ('fcb', 'LC4', '1'),
    ('fcprojw', 'LC', 'C4'), ('fcprojb', 'LC', '1'), ('lnfw', 'C', '1'), ('lnfb', 'C', '1'),
]
acts = [
    ('encoded', 'BT', 'C'), ('ln1', 'LBT', 'C'), ('ln1_mean', 'LBT', '1'), ('ln1_rstd', 'LBT', '1'),
    ('qkv', 'LBT', 'C3'), ('atty', 'LBT', 'C'), ('preatt', 'RA', 'T'), ('att', 'RA', 'T'),
    ('attproj', 'LBT', 'C'), ('residual2', 'LBT', 'C'), ('ln2', 'LBT', 'C'), ('ln2_mean', 'LBT', '1'),
    ('ln2_rstd', 'LBT', '1'), ('fch', 'LBTC4W', '32'), ('fch_gelu', 'LBTC4W', '32'), ('fcproj', 'LBT', 'C'),
    ('residual3', 'LBT', 'C'), ('lnf', 'BT', 'C'), ('lnf_mean', 'BT', '1'), ('lnf_rstd', 'BT', '1'),
    ('logits', 'BTVPW', '32'), ('probs', 'BTVPW', '32'), ('losses', 'BT', '1'),
]

def decl(prefix, table, load=False):
    out = []
    for name, r, c in table:
        v = prefix + name
        if load:
            out.append(f'  let mut {v}_h = Tensor<f32, [{r}, {c}]>::uninit();')
            out.append(f'  off = load<{r}, {c}>(&mut {v}_h, weights, off);')
        else:
            out.append(f'  let {v}_h = Tensor<f32, [{r}, {c}]>::uninit();')
        out.append(f'  let mut {v} = transfer({v}_h, Memory::GPU_HBM);')
    return '\n'.join(out)

# Gradients nothing has to zero before a step: the backward pass never reads
# these, or (preatt, att) assigns them instead of accumulating.
NO_ZERO = {'ln1_mean', 'ln1_rstd', 'ln2_mean', 'ln2_rstd', 'lnf_mean', 'lnf_rstd', 'probs', 'losses',
           'preatt', 'att'}

def zero(prefix, table, skip=()):
    return '\n'.join(f'    unsafe {{ vx_zero_f32({prefix}{n}.as_mut_ptr(), ({r} as i64) * ({c} as i64)); }}'
                     for n, r, c in table if n not in skip)

def adamw():
    return '\n'.join(
        f'    k_adamw<{r}, {c}>(&mut {n}, &d_{n}, &mut m_{n}, &mut v_{n}, lr, 0.9, 0.999, 0.00000001, wd, c1, c2);'
        for n, r, c in params)

def download_grads():
    # Each gradient tensor lands at its offset in llm.c's flat layout, then is
    # compared with the reference. wte is compared over its V real rows only.
    out = []
    for n, r, c in params:
        size = f'({r} as i64) * ({c} as i64)'
        checked = '(V as i64) * (C as i64)' if n == 'wte' else size
        out.append(f'        unsafe {{ vx_download_f32(host_grads, goff, d_{n}.as_mut_ptr(), {size}); }}')
        out.append(f'        allok = check_close(host_grads, goff, expected, grads_base + goff, {checked}, tol, "d{n}") && allok;')
        out.append(f'        goff = goff + {size};')
    return '\n'.join(out)

src = open(ROOT + '/gpu-support/train_gpu.tmpl').read()
src = src.replace('@PARAMS@', decl('', params, load=True))
src = src.replace('@GRADS@', decl('d_', params))
src = src.replace('@MS@', decl('m_', params))
src = src.replace('@VS@', decl('v_', params))
src = src.replace('@ACTS@', decl('', acts))
src = src.replace('@GRADACTS@', decl('d_', acts))
src = src.replace('@ZERO_INIT@', zero('d_', params) + '\n' + zero('m_', params) + '\n' + zero('v_', params))
src = src.replace('@ZERO_GRAD@', zero('d_', params) + '\n' + zero('d_', acts, NO_ZERO))
src = src.replace('@ADAMW@', adamw())
src = src.replace('@DOWNLOAD_GRADS@', download_grads())
open(ROOT + '/train_gpu.vx', 'w').write(src)
