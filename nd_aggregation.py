import mxnet as mx
from mxnet import nd, autograd, gluon
# import hdmedians as hd
import numpy as np
import math

def no_byz(v, f):
    pass

def marginal_median(gradients, net, lr, f = 0, byz = no_byz, factor = 0):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    byz(param_list, f, factor)
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    if sorted_array.shape[-1] % 2 == 1:
        median_nd = sorted_array[:, int((sorted_array.shape[-1]-1)/2)]
    else:
        median_nd = (sorted_array[:, (sorted_array.shape[-1]/2-1)] + sorted_array[:, (sorted_array.shape[-1]/2)]) / 2.
    # np_array = nd.concat(*param_list, dim=1).asnumpy()
    # median_nd = nd.array(np.median(np_array, axis=1))
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            param.set_data(param.data() - lr * median_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
    
def simple_mean(gradients, net, lr, f = 0, byz = no_byz, factor = 0):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    byz(param_list, f, factor)
    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            param.set_data(param.data() - lr * mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size

def score(gradient, v, f):
    if 2*f+2 >= v.shape[1]:
        f = int(math.floor((v.shape[1]-3)/2.0))
    num_neighbours = v.shape[1] - 2 - f
    sorted_distance = nd.square(v - gradient).sum(axis=0).sort()
    return nd.sum(sorted_distance[1:(1+num_neighbours)]).asscalar()

def krum(gradients, net, lr, f = 0, byz = no_byz, factor = 0):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    byz(param_list, f, factor)
    v = nd.concat(*param_list, dim=1)
    scores = nd.array([score(gradient, v, f) for gradient in param_list])
    min_idx = int(scores.argmin(axis=0).asscalar())
    krum_nd = nd.reshape(param_list[min_idx], shape=(-1,))
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            param.set_data(param.data() - lr * krum_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
