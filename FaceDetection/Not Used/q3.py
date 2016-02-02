import numpy as np
from find_c import find_c

from get_mat import get_mat


def q3():
    # The shuffling is for test examples of skin and non skin
    # (without shuffling we test only non skins)
    examples = np.random.permutation(get_mat('Skin_NonSkin.txt')) 
    powers = 2**np.arange(-5, 16, 2.0)
    learning_part = 0.7
    learning, testing = np.split(examples, [examples.shape[0]*learning_part])
    C1, errors = find_c(powers, learning)
    error = sum([r[-1] * C1(r[:-1]) < 0 for r in testing])
    error /= testing.shape[0]
    return error, np.array([powers,errors])

a , b = q3()
print(a)
print(b)