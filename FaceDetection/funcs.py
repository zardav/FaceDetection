import numpy as np


class IntImg:
    def __init__(self, mat):
        x, y = mat.shape
		#########################################################################
        # In the following lines:
        # Create matrix 'intmat' that its size is size of mat+1 (for x and for y).
        # First row and first column initialize in zero.
        # Each cell from other rows and columns initialize in sum of sub-matrix which defined by (0,0) and this cell.
        #########################################################################
        self.intmat = np.zeros((x+1, y+1))
        self.intmat[1:, 1:] = np.cumsum(np.cumsum(mat, axis=0), axis=1)

    def get_sum(self, rect):
        """
        Impelemnt of Integral Image
        :param rect: rect in the form ((start_row, end_row), (start_col, end_col))
        :return: Sum of rect
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
	"""
    The role of the functio to add rectangle around some region in picture
    :param i: (start_row, end_row)
    :param j: (start_col, end_col)
    :param rgb: Color of rectangle
    :return: New picture same img, but with rectangle around some region
    """
    img1 = np.array(img)
    i1, i2 = i
    j1, j2 = j
    img1[i1:i2, j1] = np.array(rgb)
    img1[i1:i2, j2] = np.array(rgb)
    img1[i1, j1:j2] = np.array(rgb)
    img1[i2, j1:j2] = np.array(rgb)
    return img1