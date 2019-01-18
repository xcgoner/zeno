from struct import *

def bitflip64(x,pos):
    fs = pack('d',x)
    bval = list(unpack('BBBBBBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBBBBBB', *bval)
    fnew=unpack('d',fs)
    return fnew[0]

def bitflip32(x,pos):
    fs = pack('f',x)
    bval = list(unpack('BBBB',fs))
    [q,r] = divmod(pos,8)
    bval[q] ^= 1 << r
    fs = pack('BBBB', *bval)
    fnew=unpack('f',fs)
    return fnew[0]