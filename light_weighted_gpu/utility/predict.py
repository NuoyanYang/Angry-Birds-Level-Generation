from chainer import Variable
import chainer.functions as F
from chainer import cuda
from chainer.cuda import cupy as xp
import numpy as np
def predict(vae, batch_size, z=None):
    if z is None:
        z = Variable(xp.random.normal(0, 1, (batch_size, vae.n_latent)).astype(xp.float32))
    else:
        z = Variable(xp.array(z.astype(xp.float32)))

    vae.dec.hx = F.reshape(vae.ld_h(z), (1, batch_size, 2*vae.hidden_size))
    vae.dec.cx = F.reshape(vae.ld_c(z), (1, batch_size, 2*vae.hidden_size))

    t = [[bi] for bi in [1] * batch_size]
    t = vae.make_embed_batch(t)
    ys_d = vae.dec(t, train=False)
    ys_w = [vae.h2w(y) for y in ys_d]
    name_arr_arr = []
    t = [y_each.data[-1].argmax(0) for y_each in ys_w]
    name_arr_arr.append(t)
    t = [vae.embed(xp.array([t_each], dtype=xp.int32)) for t_each in t]
    count_len = 0
    while count_len < 30:
        ys_d = vae.dec(t, train=False)
        ys_w = [vae.h2w(y) for y in ys_d]
        t = [y_each.data[-1].argmax(0) for y_each in ys_w]
        name_arr_arr.append(t)
        t = [vae.embed(xp.array([t_each], dtype=xp.int32)) for t_each in t]
        count_len += 1

    lines = []
    tenti = []
    for i in range(len(name_arr_arr)):
        name = name_arr_arr[i][0].item()
        tenti.append(name)

    tenti = [tenti]

    for name in tenti:
        name = [vae.vocab.itos(nint) for nint in name]
        if "</s>" in name:
            line = name[:name.index("</s>")]
            lines.append(line)
        else:
            lines.append(name)

    return lines
