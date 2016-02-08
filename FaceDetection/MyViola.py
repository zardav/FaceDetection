import numpy as np
from scipy import ndimage

import funcs
from AbstractClassifier import AbstractClassfier
from MySvm import MySvm


class _Feature:
    def __init__(self, trans, plus, minus):
        self.plus = plus
        self.minus = minus
        self.trans = trans

    def calc(self, window_start):
        i, j = window_start
        result = 0
        for (x, y) in self.plus:
            r = ((i+x[0], i+x[1]), (j+y[0], j+y[1]))
            result += self.trans.intimg.get_sum(r)
        for (x, y) in self.minus:
            r = ((i+x[0], i+x[1]), (j+y[0], j+y[1]))
            result -= self.trans.intimg.get_sum(r)
        return result


class _Transform:
    def __init__(self, func):
        self.func = func
        self.mat = None
        self.intimg = None

    def change_image(self, img):
        self.mat = self.func(img).astype('float64')
        self.intimg = funcs.IntImg(self.mat)


class _SubClassifier(AbstractClassfier):
    def from_list(self, mat):
        self.svm.from_list(mat)

    def classify_vec(self, rect, axis=-1):
        return self.svm.classify_vec(self.calc_features(rect), axis)

    def to_list(self):
        return self.svm.to_list()

    def __init__(self, features, rejecter=False):
        super().__init__()
        self.svm = MySvm()
        self._features = features
        self._row_len = len(features) + 1
        self.examples = np.empty((0, self._row_len))
        self.is_rejecter = rejecter

    def add_current_image(self, y):
        row = np.empty(self._row_len)
        row[-1] = y
        row[:-1] = self.calc_features()
        self.examples = np.vstack((self.examples, row))

    def calc_features(self, rect=None):
        start = (0, 0) if rect is None else (rect[0][0], rect[1][0])
        return np.array([f.calc(start) for f in self._features])

    def learn(self, d=None):
        examples = np.copy(self.examples)
        x, y = examples[:, :-1], examples[:, -1]
        if d is None and self.is_rejecter:
            d = np.fromiter((30 if s == 1 else 1 for s in y), dtype=int)
        self.svm.learn(x, y, d, c_arr=2**np.arange(-5, 15, 2.0),
                       epoch=15, cross_validation_times=7, learning_part=0.7)

    def classify(self, rect):
        return self.svm.classify(self.calc_features(rect))

    def valuefy(self, rect):
        return self.svm.valuefy(self.calc_features(rect))


class MyViolaClassifier(AbstractClassfier):
    def __init__(self):
        super().__init__()
        self.classifiers = []
        self.sorted_classifiers = []
        self.plus_minus_rects = []
        self.transform_lists = []
        self.img = None
        self.add_basic_transforms()
        self.add_basic_features()
        self.add_basic_classifiers()
        self.num_examples = 0
        self.weight_vec = np.empty(0)
        self.iv_weight = np.empty(0)

    def change_image(self, img):
        self.img = img
        self.apply_image()

    def apply_image(self):
        for t_list in self.transform_lists:
            for t in t_list:
                t.change_image(self.img)

    def add_basic_transforms(self):
        rgb = [_Transform(lambda im, k=i: im[..., k]) for i in range(3)]
        laplace = [_Transform(lambda im, k=i: ndimage.filters.laplace(im)[..., k]) for i in range(3)]
        sob = ndimage.sobel
        sob1 = [_Transform(lambda im, k=i: sob(im[..., k], j)) for i in range(3) for j in range(2)]
        sob2 = [_Transform(lambda im, k=i: sob(sob(im[..., k], j1), j2))
                for i in range(3) for j1 in range(2) for j2 in range(2)]
        grad = [_Transform(lambda im, k=i: np.arctan2(sob(im[..., k], 0), sob(im[..., k], 1)))
                for i in range(3)] + [_Transform(
                lambda im, k=i: np.hypot(sob(im[..., k], 0), sob(im[..., k], 1))) for i in range(3)]
        self.transform_lists += [laplace, grad, rgb, sob1, sob2]

    def add_basic_features(self):
        def add(l): self.plus_minus_rects.append(l)
        add([([((0, 69), (0, 69))], [])])  # whole image
        add([([((53, 67), (18, 52)), ((53, 67), (18, 52))], [((50, 69), (10, 60))])])  # mouth vs around mouth
        add([([((34, 52), (27, 43))], [((34, 52), (10, 26)), ((34, 52), (44, 60))]),  # nose vs sides of nose
             ([((17, 35), (7, 63))], [((36, 52), (7, 63))])])  # eyes vs nose
        add([([((17, 35), (7, 63))], [((4, 17), (7, 63))])])  # eyes vs forehead
        add([([((17, 35), (7, 31)), ((17, 35), (39, 63))], []),  # eyes only
             ([((34, 52), (22, 48))], [])])  # nose only
        add([([((51, 68), (18, 52))], [])])  # mouth only
        add([([((0, 69), (0, 34))], [((0, 69), (50, 69))])])  # horizontal symmetric
        add([([((0, 69), (0, 69))], [((0, 69), (4, 65)), ((0, 69), (4, 65))])])  # horizontal edges
        add([([((0, 34), (0, 69))], [((35, 69), (0, 69))])])  # vertical symmetric
        add([([((0, 20), (0, 5)), ((49, 69), (0, 5)), ((0, 20), (64, 69)), ((49, 69), (64, 69))], [])])  # border sides
        add([([((0, 69), (0, 7))], [((0, 69), (8, 15))]), ([((0, 69), (62, 69))], [((0, 69), (55, 62))])])  # sides -in
        add([([((0, 69), (14, 56))], [((0, 69), (29, 41)), ((0, 69), (29, 41))])])  # horizontal middle
        add([([((0, 18), (10, 60))], [])]) # forehead only
        add([([((54, 69), (10, 60))], [((38, 53), (10, 60))])])  # mouth vs nose
        add([([((10, 50), (14, 28)), ((10, 50), (42, 56))], [((10, 50), (28, 42))])]) # horizontal middle, eyes and nose


    def add_basic_classifiers(self):
        for t_list in self.transform_lists:
            for f_list in self.plus_minus_rects:
                features = []
                for t in t_list:
                    for f in f_list:
                        features.append(_Feature(t, f[0], f[1]))
                self.classifiers.append(_SubClassifier(features))
                self.classifiers.append(_SubClassifier(features, rejecter=True))

    def add_examples(self, imgs, y):
        for img in imgs:
            self.change_image(img)
            self.num_examples += 1
            for classifier in self.classifiers:
                classifier.add_current_image(y)

    def learn(self):
        for classifier in self.classifiers:
            classifier.learn()
        self.sort_classifiers()

    def sort_classifiers(self):
        self.sorted_classifiers = [c for c in self.classifiers if c.svm.detection_rate > 0.1]
        self.sorted_classifiers = sorted(self.sorted_classifiers, key=lambda c: c.svm.detection_rate, reverse=True)
        self.weight_vec = np.fromiter(((1 / (0.01 + c.svm.fp_error)) for c in self.sorted_classifiers),
                                      dtype='float64')
        self.weight_vec /= self.weight_vec.sum()
        self.iv_weight = np.cumsum(self.weight_vec)

    def to_list(self):
        return [c.to_list() for c in self.classifiers]

    def from_list(self, list_):
        for c, l in zip(self.classifiers, list_):
            if not isinstance(l, list):
                raise ValueError('MyViolaClassifier.from_list: list_ has to be list of lists')
            c.from_list(l)
        self.sort_classifiers()

    def classify(self, rect):
        return 1 if self.valuefy(rect) > 0 else -1

    def valuefy(self, rect):
        sum_ = 0
        p = 0.001
        i = 0
        m = len(self.sorted_classifiers)
        def mult_prob(p1, p2):
            t_val, f_val = p1*p2, (1-p1)*(1-p2)
            return t_val / (t_val + f_val)
        while i < m:
            rej = self.sorted_classifiers[i]
            cur = rej.valuefy(rect)
            if cur < 0:
                p *= 1 - rej.svm.detection_rate
                if p < 0.00001:
                    return -1
            else:
                sum_ += cur * self.weight_vec[i]
                p = mult_prob(p, 1 - rej.svm.fp_error)
                if p > 0.99:
                    return sum_ / self.iv_weight[i]
            i += 1
        return sum_

