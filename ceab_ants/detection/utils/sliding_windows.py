
import numpy as np


AFTER = 'after'
BEFORE = 'before'
BOTH = 'both'

CONSTANT_1 = 'constant_1'

def padForSlide(frame, size, overlap, mode=CONSTANT_1, edges=AFTER):
    kwargs = dict()
    if mode == CONSTANT_1:
        mode = 'constant'
        kwargs = dict(constant_values=1)

    size = np.asarray(size)
    h, w = frame.shape[:2]

    # Distancia entre principios de ventana. La zona que añade una ventana respecto a solo una adiacente
    strides = (size * (1 - overlap)).astype(int)

    # Si el espacio que le falta a la ultima ventana equivale al espacio nuevo, la ventana anterior es la última
    # Si el espacio es multiple de los strides, la ultima ventana necesita un espacio equivalente al solape (size % stride)
    pad_h = (size[0] - (h % strides[0])) % strides[0]
    pad_w = (size[1] - (w % strides[1])) % strides[1]

    if (pad_h == 0) and (pad_w == 0):
        return frame # View better than copy
    
    if edges == AFTER:
        pad = [(0, pad_h), (0, pad_w)]
    elif edges == BEFORE:
        pad = [(pad_h, 0), (pad_w, 0)]
    elif edges == BOTH:
        pad =[(pad_h // 2, pad_h // 2 + pad_h % 2), (pad_w // 2, pad_w // 2 + pad_w % 2)]
    else:
        raise NotImplementedError()
    
    pad = pad + [(0, 0) for _ in frame.shape[2:]]

    return np.pad(frame, pad, mode=mode, **kwargs)

def sliceFrame(frame, size, overlap=0.2, batch=False):

    if isinstance(size, int) : size = (size, size)
    
    assert len(size) == 2 and isinstance(size[0], int) and isinstance(size[1], int)
    assert overlap < 1

    h, w = frame.shape[:2]
    size = np.asarray(size)
    strides = (size * (1 - overlap)).astype(int)

    pad_frame = padForSlide(frame, size, overlap, mode=CONSTANT_1, edges=AFTER)

    points = np.stack(np.meshgrid(np.arange(0, h, strides[0]), np.arange(0, w, strides[1]))).reshape(2, -1)

    if batch:
        crops = [pad_frame[x : x + size[0], y : y + size[1]] for x, y in zip(*points)]
        return crops, points.T
    else:
        crops = {(x, y) : pad_frame[x : x + size[0], y : y + size[1]] for x, y in zip(*points)}
        return crops
        