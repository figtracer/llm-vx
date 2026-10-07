# Generates train_gpu.vx from train_gpu.tmpl: the template holds the logic, and
# the tables below expand into the per-tensor declarations, zeroing, AdamW and
# gradient checks.
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params = [  # name, rows, cols; the gradient's rows and cols where they differ
    ('wte', 'VPCW', 'EW', 'VP', 'C'), ('wpe', 'MAXTCW', 'EW', 'MAXT', 'C'), ('ln1w', 'LC', '1'), ('ln1b', 'LC', '1'),
    ('qkvw', 'LC3CW', 'EW', 'LC3', 'C'), ('qkvb', 'LC3', '1'), ('attprojw', 'LCCW', 'EW', 'LC', 'C'),
    ('attprojb', 'LC', '1'), ('ln2w', 'LC', '1'), ('ln2b', 'LC', '1'), ('fcw', 'LC4CW', 'EW', 'LC4', 'C'),
    ('fcb', 'LC4', '1'), ('fcprojw', 'LC4CW', 'EW', 'LC', 'C4'), ('fcprojb', 'LC', '1'), ('lnfw', 'C', '1'),
    ('lnfb', 'C', '1'),
]
# The weights and their moments are EW wide, so AdamW reads them in one go; the
# gradients keep the rows that cuBLAS and the encoder kernels write.
grads = [(p[0], *p[3:5]) if len(p) == 5 else p for p in params]
params = [p[:3] for p in params]
acts = [
    ('encoded', 'BT', 'C'), ('ln1', 'LBT', 'C'), ('ln1_mean', 'LBT', '1'), ('ln1_rstd', 'LBT', '1'),
    ('qkv', 'LBT', 'C3'), ('atty', 'LBT', 'C'), ('preatt', 'RAW', 'AW'), ('att', 'RAW', 'AW'),
    ('attproj', 'LBT', 'C'), ('residual2', 'LBT', 'C'), ('ln2', 'LBT', 'C'), ('ln2_mean', 'LBT', '1'),
    ('ln2_rstd', 'LBT', '1'), ('fch', 'LBTC4W', 'EW'), ('fch_gelu', 'LBTC4W', 'EW'), ('fcproj', 'LBT', 'C'),
    ('residual3', 'LBT', 'C'), ('lnf', 'BT', 'C'), ('lnf_mean', 'BT', '1'), ('lnf_rstd', 'BT', '1'),
    ('logits', 'BTVPW', 'EW'), ('probs', 'BTVPW', 'EW'), ('losses', 'BT', '1'),
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

def zero(prefix, table):
    return '\n'.join(f'    unsafe {{ vx_zero_f32({prefix}{n}.as_mut_ptr(), ({r} as i64) * ({c} as i64)); }}'
                     for n, r, c in table)

def adamw():
    return '\n'.join(
        f'    k_adamw<{r}, {c}, {gr}, {gc}>(&mut {n}, &d_{n}, &mut m_{n}, &mut v_{n}, lr, 0.9, 0.999, 0.00000001, wd, c1, c2);'
        for (n, r, c), (_, gr, gc) in zip(params, grads))

def download_grads():
    # Each gradient tensor lands at its offset in llm.c's flat layout, then is
    # compared with the reference. wte is compared over its V real rows only.
    out = []
    for n, r, c in grads:
        size = f'({r} as i64) * ({c} as i64)'
        checked = '(V as i64) * (C as i64)' if n == 'wte' else size
        out.append(f'        unsafe {{ vx_download_f32(host_grads, goff, d_{n}.as_mut_ptr(), {size}); }}')
        out.append(f'        allok = check_close(host_grads, goff, expected, grads_base + goff, {checked}, tol, "d{n}") && allok;')
        out.append(f'        goff = goff + {size};')
    return '\n'.join(out)

src = open(ROOT + '/gpu-support/train_gpu.tmpl').read()
src = src.replace('@PARAMS@', decl('', params, load=True))
src = src.replace('@GRADS@', decl('d_', grads))
src = src.replace('@MS@', decl('m_', params))
src = src.replace('@VS@', decl('v_', params))
src = src.replace('@ACTS@', decl('', acts))
src = src.replace('@GRADACTS@', decl('d_', acts))
src = src.replace('@ZERO_INIT@', zero('d_', grads) + '\n' + zero('m_', params) + '\n' + zero('v_', params))
# Only parameter gradients accumulate. Each activation gradient is set by the
# one kernel or cuBLAS call that writes it, so none needs zeroing, except that
# d_attproj and d_fcproj are zeroed although nothing reads them: they are copies
# of d_residual2 and d_residual3 in llm.c's CPU code, which this program does not
# make, and zeroing them each step keeps them live through the step, so the
# memory Vx counts stays llm.c's.
UNUSED = {'attproj', 'fcproj'}
src = src.replace('@ZERO_GRAD@', zero('d_', grads) + '\n' + zero('d_', [a for a in acts if a[0] in UNUSED]))
src = src.replace('@ADAMW@', adamw())
src = src.replace('@DOWNLOAD_GRADS@', download_grads())
open(ROOT + '/train_gpu.vx', 'w').write(src)
