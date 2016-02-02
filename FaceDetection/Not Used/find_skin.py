import numpy as np
from find_c import find_c
from scipy import misc

from get_mat import get_mat


def skin_mat():
    examples = np.random.permutation(get_mat('Skin_NonSkin.txt'))
    powers = 2**np.arange(-5, 16, 2.0)
    classifier, errors = find_c(powers, examples)

    img = misc.imread('image_0009.jpg')
    skin = np.array(img[..., 0])
    m, n = skin.shape
    for i in range(m):
        for j in range(n):
            r, g, b = img[i, j]
            s = classifier(np.array([b, g, r]))
            skin[i, j] = 255 if s > 0 else 0
    return skin

skin = skin_mat()
