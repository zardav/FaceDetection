import numpy as np


def get_mat(file_name):
    mat = np.loadtxt(file_name)
    mat[:,-1] = 2*(1-mat[:,-1])+1  # replace all '2' with '-1'. ones not affected.
    return mat