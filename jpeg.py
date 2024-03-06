

import numpy as np
import scipy.fftpack as fftpack

from huffman import huffman_coding
from utils import color_space_conversion, downsample, to_blocks, quantize, \
    zigzag, run_length_encoding, save_to_jpeg, bits_required


def dct(arr):
    """
    discrete cosine transform, normalize to center at 0, then use scipy

    Parameters:
        arr (np.ndarray): with shape (B, N * N)
    """
    arr = arr - 127
    arr = fftpack.dct(fftpack.dct(arr.T, norm='ortho').T, norm='ortho')

    return arr


def entropy_encoding(arr):
    arr = run_length_encoding(arr)
    binary, tree = huffman_coding(arr)

    return binary, tree


def jpeg_conversion(arr):
    """
    five steps:

    1. color space conversion
    2. down-sampling
    3. division into 8x8 blocks
    4. normalize & forward dct (discrete cosine transform)
    5. quantization & turn into 1d using zigzag pattern
    6. entropy encoding (run length encoding & huffman coding)

    """
    arr = color_space_conversion(arr)
    arr = downsample(arr)
    blocks = to_blocks(arr)
    weights = dct(blocks)
    weights = quantize(weights)
    weights = zigzag(weights)

    # create dc and ac as required by the jpeg format
    dc = weights[..., 0].reshape(-1, 3)
    ac = weights[..., 1:].reshape(-1, 3, 63).transpose()

    # create huffman coding
    huffmans = {
        'dc_y': huffman_coding(np.vectorize(bits_required)(dc[:, 0])),
        'ac_y': huffman_coding(np.vectorize(bits_required)(dc[:, 1:].flat)),
        'dc_c': huffman_coding(run_length_encoding(arr)),
        'ac_c': huffman_coding(run_length_encoding(arr))
    }

    save_to_jpeg("results.jpg", dc, ac, huffmans)


if __name__ == "__main__":

    from utils import read_img

    arr = read_img("img/apple.jpg")
    jpeg_conversion(arr)

    # arr = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1])
    # print(run_length_encoding(arr))

    # img = read_img("apple.jpg")
    # img = downsample(img)
    # img = pad_for_blocks(img)
    #
    # print(to_blocks(img).shape)
    #

