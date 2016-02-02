import numpy as np
import math
from scipy import misc
import funcs
from MyViola import MyViola
import Svm


class CropFace:
    def __init__(self, img):
        self._img = img
        self._scale = np.arange(.2, .35, .02)

    def _find_nearest(self, shape, svm):
        mv = MyViola()
        max_ = -math.inf
        res_i = (0, 0)
        res_j = (0, 0)
        m, n = shape
        for scl in self._scale:
            img = misc.imresize(self._img, scl)
            mv.change_image(img)
            x, y = img.shape[:2]
            if x < m or y < n:
                continue
            for i, j in funcs.iter_shape((x, y), shape):
                val = svm.valuefy(mv.calc_features((i,j)))
                if val > max_:
                    max_ = val
                    res_i = (int(i[0] / scl), int(i[1] / scl))
                    res_j = (int(j[0] / scl), int(j[1] / scl))
        return res_i, res_j
