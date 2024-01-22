import numpy as np
from chainer import Variable
import chainer.functions as F
def predict(vae, batch_size, z=None):
    if z is None:
        z = Variable(np.random.normal(0, 1, (batch_size, vae.n_latent)).astype(np.float32))
    else:
        z = Variable(z.astype(np.float32))

    vae.dec.hx = F.reshape(vae.ld_h(z), (1, batch_size, 2*vae.hidden_size))
    vae.dec.cx = F.reshape(vae.ld_c(z), (1, batch_size, 2*vae.hidden_size))

    t = [[bi] for bi in [1] * batch_size]
    t = vae.make_embed_batch(t)
    ys_d = vae.dec(t, train=False)
    ys_w = [vae.h2w(y) for y in ys_d]
    name_arr_arr = []
    t = [y_each.data[-1].argmax(0) for y_each in ys_w]
    name_arr_arr.append(t)
    t = [vae.embed(np.array([t_each], dtype=np.int32)) for t_each in t]
    count_len = 0
    while count_len < 30:
        ys_d = vae.dec(t, train=False)
        ys_w = [vae.h2w(y) for y in ys_d]
        t = [y_each.data[-1].argmax(0) for y_each in ys_w]
        name_arr_arr.append(t)
        t = [vae.embed(np.array([t_each], dtype=np.int32)) for t_each in t]
        count_len += 1

    tenti = np.array(name_arr_arr, dtype=np.int32).T

    lines = []
    for name in tenti:
        name = [vae.vocab.itos(nint) for nint in name]
        if "</s>" in name:
            line = name[:name.index("</s>")]
            lines.append(line)
        else:
            lines.append(name)
    return lines
