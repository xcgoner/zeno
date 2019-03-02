import mxnet as mx
from mxnet import nd

def no_byz(v, f, factor):
    pass

def median_attack(v, f, factor):
    v_correct = v[f:]
    byz_v = -100 * nd.sum(nd.concat(*v_correct, dim=1), axis=-1)
    for i in range(f):
        v[i] = byz_v

def krum_attack(v, f, factor):
    v_correct = v[f:]
    byz_v = -1 * nd.sum(nd.concat(*v_correct, dim=1), axis=-1)
    for i in range(f):
        v[i] = byz_v

def signflip_attack(v, f, factor):
    v_correct = v[f:]
    byz_v = -factor * nd.sum(nd.concat(*v_correct, dim=1), axis=-1)
    for i in range(f):
        v[i] = byz_v


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