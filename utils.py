

import numpy as np
import cv2


def read_img(filename):
    """read jpeg image file"""
    return cv2.imread(filename)


def pad_for_blocks(arr, N=8):
    """pad for no remainders of N, shape has to be (H, W, C)"""
    (H, W, C) = arr.shape
    temp = np.zeros((H + N - (H % N), W + N - (W % N), C), dtype=arr.dtype)
    temp[:arr.shape[0], :arr.shape[1], :] = arr
    return temp


def to_blocks_(arr, N=8):
    """
    split arr into NxN blocks
    from (H, W) to (H//N * W//N, NxN)
    """
    tiles = [arr[x:x+N,y:y+N] for x in range(0,arr.shape[0],N) for y in range(0,arr.shape[1],N)]
    return np.array(tiles).reshape(-1, N*N)


def to_blocks(arr, N=8):
    """
    split arr into NxN blocks
    from (H, W, C) to (H//N * W//N * C, NxN)
    """
    arr = pad_for_blocks(arr)
    (H, W, C) = arr.shape

    tiles1 = to_blocks_(arr[..., 0])
    tiles2 = to_blocks_(arr[..., 1])
    tiles3 = to_blocks_(arr[..., 2])

    tiles = np.zeros(((H//N) * (W//N) * C, N*N))
    tiles[0::3, :] = tiles1
    tiles[1::3, :] = tiles2
    tiles[2::3, :] = tiles3

    return tiles.reshape((H//N) * (W//N) * C, N, N)


def downsample(arr):
    """down-sample from (H, W, C) to (H//2, W//2, C)"""
    return cv2.resize(arr, (0, 0), fx=0.5, fy=0.5)


def color_space_conversion(arr):
    """converts rgb format to ycrcb format"""
    return cv2.cvtColor(arr, cv2.COLOR_BGR2YCR_CB)


def quantize(arr, N=8):  # , chrominance, luminance):
    """
    quantization: divide the weight matrix by a precalculated quantization matrix

    Parameters:
        arr (np.ndarray): assume shape should be (B, C, N, N) or (B * C, N, N)

    Returns:
        arr (np.ndarray): with shape (B * C, N, N)
    """
    print(arr.shape)

    chrominance = np.array([
        [10,  8,  9,  9,  9,  8, 10,  9],
        [ 9,  9, 10, 10, 10, 11, 12, 17],
        [13, 12, 12, 12, 12, 20, 16, 16],
        [14, 17, 18, 20, 23, 23, 22, 20],
        [25, 25, 25, 25, 25, 25, 25, 25],
        [25, 25, 25, 25, 25, 25, 25, 25],
        [25, 25, 25, 25, 25, 25, 25, 25],
        [25, 25, 25, 25, 25, 25, 25, 25],
    ]).reshape(1, 1, N, N)

    luminance = np.array([
        [ 6,  4,  4,  6, 10, 16, 20, 24],
        [ 5,  5,  6,  8, 10, 23, 24, 22],
        [ 6,  5,  6, 10, 16, 23, 28, 22],
        [ 6,  7,  9, 12, 20, 35, 32, 25],
        [ 7,  9, 15, 22, 27, 44, 41, 31],
        [10, 14, 22, 26, 32, 42, 45, 37],
        [20, 26, 31, 35, 41, 48, 48, 40],
        [29, 37, 38, 39, 45, 40, 41, 40],
    ]).reshape(1, 1, N, N)

    arr = arr.reshape(-1, 3, N, N)

    arr[:, 0, :, :] = arr[:, 0, :, :] / luminance
    arr[:, 1, :, :] = arr[:, 1, :, :] / chrominance
    arr[:, 2, :, :] = arr[:, 2, :, :] / chrominance

    arr = arr.reshape(-1, N, N)

    return np.round(arr).astype(np.int32)


def zigzag_(arr):
    """converts a 2d matrix into a 1d array using a zigzag pattern according to jpeg"""
    arr = np.concatenate([np.diagonal(arr[::-1,:], i)[::(2*(i % 2)-1)] for i in range(1-arr.shape[0], arr.shape[0])])
    return arr


def zigzag(arr):
    """converts a 2d matrix into a 1d array using a zigzag pattern according to jpeg"""
    return np.array([zigzag_(a) for a in arr])


def run_length_encoding(arr):
    """
    converts a list of integers to a list of integers with number of repeats

    Example:
        "1110000011111" -> "130515"
    """

    ans = ""
    count = 0
    for i, n in enumerate(arr):
        count += 1

        if i+1 == len(arr):
            ans += str(n) + str(count)
            continue

        if arr[i+1] == n:
            continue

        ans += str(n) + str(count)
        count = 0

    return ans


def show_img(arr):
    # save it as well
    cv2.imwrite("results.png", arr)

    # show it
    cv2.imshow("image", arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def uint_to_binstr(n, size):
    return bin(n)[2:][-size:].zfill(size)


def bits_required(n):
    """find position of highest 1 bit"""
    n = abs(n)
    result = 0
    print(n)
    while n > 0:
        n >>= 1
        print(n)
        result += 1
    return result


def save_to_jpeg(filename, binary, tree, n_blocks):
    """
    creates jpeg file with filename and convert all necessary python objects
    into binary to be written into that file as required by jpeg

    Parameters:
        filename (string): filename
        binary (string): string of zeros and ones of huffman encoding
        tree (NodeTree): root node of huffman tree
        n_blocks (int): number of blocks the image has

    """
    f = open(filename, 'w')

    # for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
    #
    #     # 16 bits for 'table_size'
    #     f.write(uint_to_binstr(len(tables[table_name]), 16))
    #

    # 32 bit for 'n_blocks'
    f.write(uint_to_binstr(n_blocks, 32))

    #


