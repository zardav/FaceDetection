import numpy as np
from AbstractClassifier import AbstractClassfier


class Svm(AbstractClassfier):
    def __init__(self):
        self._w = np.array([])
        self._mins = np.array([])
        self._maxs = np.array([])
        self.false_positive_loss = 1
        self.false_negative_loss = 1
        self.error = 0
        self.simple_error = 0

    def _sub_gradient_loss(self, example, W, c, n):
        (x, y) = example[:-1], example[-1]
        grad_loss = W / n
        if 1 - self._loss_value(y)*y * W.dot(x) > 0:
            grad_loss -= c*self._loss_value(y)*y * x
        return grad_loss

    def _loss_value(self, y):
        return self.false_negative_loss if y == 1 else self.false_positive_loss

    def _svm_c(self, examples, c, epoch):
        """

        :param examples:
        :param c:
        :return:
        """
        num_ex = examples.shape[0]
        alpha = 1
        w_vec = np.zeros(examples.shape[1]-1)
        t = 0
        for _ in range(epoch * num_ex):
            r = examples[np.random.randint(num_ex)]
            t += 1
            w_vec -= alpha/t * self._sub_gradient_loss(r, w_vec, c, num_ex)
        return w_vec

    def learn(self, examples, c_arr=None, epoch=3, learning_part=0.7, cross_validation_times=5):
        """

        :param learning_part:
        :param examples:
        :param cross_validation_times:
        :param epoch:
        :param x_mat: Vector X of vectors of features
        :param y_vec:
        :param c_arr:
        """
        if c_arr is None:
            c_arr = 2**np.arange(-5, 16, 2.0)

        n = examples.shape[0]
        middle = round(n*learning_part) if learning_part > 0 else -1
        # normalize
        x_mat = examples[:, :-1]
        mins = x_mat.min(axis=0)
        maxs = x_mat.max(axis=0) - mins
        x_mat[:] = (x_mat - mins) / maxs
        c_len = c_arr.shape[0]
        errors = np.zeros(c_len)
        for j in range(c_len):
            c = c_arr[j]
            error = 0
            for _ in range(cross_validation_times):
                shuffled = np.random.permutation(examples)
                learnings, testings = np.split(shuffled, [middle])
                w_vec = self._svm_c(learnings, c, epoch)
                error = sum([(r[-1] * w_vec.dot(r[:-1]) < 0) * self._loss_value(r[-1]) for r in testings])
            errors[j] += error/testings.shape[0]
        errors /= cross_validation_times
        result_c = c_arr[np.argmin(errors)]
        w_vec = self._svm_c(examples, result_c, epoch)
        #ending
        self._w = w_vec
        self._mins = mins
        self._maxs = maxs
        self.error = sum([(r[-1] * w_vec.dot(r[:-1]) < 0) * self._loss_value(r[-1]) for r in examples]) / n
        self.simple_error = sum([(r[-1] * w_vec.dot(r[:-1]) < 0) for r in examples]) / n

    def to_list(self):
        return [self._w, self._mins, self._maxs, self.error, self.simple_error]

    def from_list(self, list_):
        if len(list_) != 5:
            raise ValueError('from_list: len(list_) has to be 5')
        self._w, self._mins, self._maxs, self.error, self.simple_error = list_

    def classify(self, x):
        return self._w.dot((x - self._mins) / self._maxs) > 0

    def classify_vec(self, vec, axis=-1):
        return np.apply_along_axis(self.classify, axis, vec)

    def valuefy(self, x):
        return self._w.dot((x - self._mins) / self._maxs)

    def valuefy_vec(self, vec, axis=-1):
        return np.apply_along_axis(self.valuefy, axis, vec)

