import bitflip
import mxnet as mx
from mxnet import nd

def no_byz(v, f, factor):
    pass

def gaussian_attack(v, f, factor):
    for i in range(f):
        v[i] = mx.nd.random.normal(0, 200, shape=v[i].shape)

def omniscient_attack(v, f, factor):
    byz_v = -1e10 * nd.sum(nd.concat(*v, dim=1), axis=-1)
    for i in range(f):
        v[i] = byz_v

def bitflip_attack(v, f, factor):
    for i in range(500):
        b = bitflip.bitflip32(v[0][i].asscalar(), 22)
        b = bitflip.bitflip32(b, 29)
        b = bitflip.bitflip32(b, 30)
        b = bitflip.bitflip32(b, 31)
        v[0][i] = b
    for i in range(f):
        if i > 0:
            v[i][:] = v[0]

def signflip_attack(v, f, factor):
    for i in range(f):
        if i > 0:
            v[i][:] = -v[0]
    v[0][:] = -v[0]

# def bitflip_attack(v, f):
#     for i in range(f):
#         b = bitflip.bitflip32(v[i][i].asscalar(), 22)
#         b = bitflip.bitflip32(b, 29)
#         b = bitflip.bitflip32(b, 30)
#         b = bitflip.bitflip32(b, 31)
#         v[i][i] = b

# def omniscient_attack(v, f, factor):
#     for i in range(f):
#         v[i] *= (-factor)

# def omniscient0_attack(v, f, factor):
#     byz_v = (-factor) * nd.mean(nd.concat(*v, dim=1), axis=-1, keepdims=True)
#     for i in range(f):
#         v[i] = byz_v