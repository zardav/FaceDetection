import numpy as np


def max_sub_arr(arr):
    cur_sum = max_sum = arr[0]
    max_start = max_end = cur_start = cur_end = 0
    for i in range(1, len(arr)):
        cur_sum += arr[i]
        cur_end = i
        if cur_sum < arr[i]:
            cur_start, cur_sum = i, arr[i]
        if cur_sum > max_sum:
            max_sum, max_start, max_end = cur_sum, cur_start, cur_end
    return max_sum, max_start, max_end


def max_sub_mat(mat):
    max_sum = mat[0, 0]
    m, n = mat.shape
    i_start = i_end = j_start = j_end = 0
    for p in range(n):
        rows_part_sum = np.zeros(m)
        for q in range(p, n):
            rows_part_sum += mat[:, q]
            max_, start_, end_ = max_sub_arr(rows_part_sum)
            if max_ > max_sum:
                max_sum = max_
                j_start, j_end = p, q
                i_start, i_end = start_, end_
    return max_sum, j_start, j_end, i_start, i_end


class IntImg:
    def __init__(self, mat):
        x, y = mat.shape
        self.intmat = np.zeros((x+1, y+1))
        self.intmat[1:, 1:] = np.cumsum(np.cumsum(mat, axis=0), axis=1)

    def get_sum(self, rect):
        """
        :param rect: rect in the form ((start_row, end_row), (start_col, end_col))
        :return:
        """
        i, j = rect
        i1, i2 = i
        j1, j2 = j
        result = self.intmat[i2+1, j2+1] - self.intmat[i2+1, j1] - self.intmat[i1, j2+1] + self.intmat[i1, j1]
        return result


def iter_shape(outer_shape, inner_shape, step):
    x, y = outer_shape
    m, n = inner_shape
    return (((i, i+m), (j, j+n)) for i in range(0, x-m, step) for j in range(0, y-n, step))


def implusrect(img, i, j, rgb):
    img1 = np.array(img)
    i1, i2 = i
    j1, j2 = j
    img1[i1:i2, j1] = np.array(rgb)
    img1[i1:i2, j2] = np.array(rgb)
    img1[i1, j1:j2] = np.array(rgb)
    img1[i2, j1:j2] = np.array(rgb)
    return img1


def ada_boost(classifier_list, examples, ):
    _T = len(classifier_list)
    m = examples.shape[0]
    _D
