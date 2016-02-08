import numpy as np
from AbstractClassifier import AbstractClassfier


class MySvm(AbstractClassfier):
    def __init__(self):
        self._w = np.array([])
        self._mins = np.array([])
        self._maxs = np.array([])
        self._divisor = 1
        self.alpha = 1
        self._b = 0
        self.weighted_error = 0
        self.fn_error = 0
        self.fp_error = 0
        self.simple_error = 0

    @staticmethod
    def _sub_gradient_loss(x, y, d, w_vec, c, n):
        grad_loss = 0
        if 1 - d*y * w_vec.dot(x) > 0:
            grad_loss = w_vec / n - c*d*y * x
        else:
            grad_loss = w_vec / n
        return grad_loss

    def _svm_c(self, x_2d_vec, y_1d_vec, d_1d_vec, c, epoch):
        """

        :param examples:
        :param c:
        :return:
        """
        num_ex = x_2d_vec.shape[0]
        w_vec = np.zeros(x_2d_vec.shape[1])
        t = 1
        for _ in range(epoch * num_ex):
            i = np.random.randint(num_ex)
            w_vec -= self.alpha/t * self._sub_gradient_loss(x_2d_vec[i], y_1d_vec[i], d_1d_vec[i], w_vec, c, num_ex)
            t += 1
        return w_vec

    def learn(self, x, y, d=None, c_arr=None, epoch=3, learning_part=0.7, cross_validation_times=5):
        """

        :param learning_part:
        :param examples:
        :param cross_validation_times:
        :param epoch:
        :param c_arr:
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError('Size of x and y has to be same')
        if x.shape[0] == 0:
            raise ValueError('Zero examples')
        if c_arr is None:
            c_arr = 2**np.arange(-5, 16, 2.0)
        if d is None:
            d = np.ones_like(y)

        n = y.shape[0]
        x = np.c_[x, x**2]
        middle = round(n*learning_part) if learning_part > 0 else -1
        # normalize
        mins = x.min(axis=0)
        maxs = x.max(axis=0) - mins
        maxs[maxs == 0] = 1
        x[:] = (x - mins) / maxs
        x = np.c_[x, np.ones(n)]
        c_len = c_arr.shape[0]
        errors = np.zeros((c_len, 4))  # simple, weighted, fp, fn
        for j in range(c_len):
            c = c_arr[j]
            for _ in range(cross_validation_times):
                shuffled_i = np.random.permutation(n)
                learnings_i, testings_i = shuffled_i[:middle], shuffled_i[middle:]
                w_vec = self._svm_c(x[learnings_i], y[learnings_i], d[learnings_i], c, epoch)
                answers = [(1 if w_vec.dot(x[i]) > 0 else -1) for i in testings_i]
                range_ = range(len(answers))
                error_list = [(y[testings_i[i]], d[testings_i[i]]) for i in range_ if answers[i] != y[testings_i[i]]]
                errors[j] += self.get_all_errors(answers, error_list)
        errors /= cross_validation_times
        index_min = errors[:, 0].argmin()
        result_c = c_arr[index_min]
        w_vec = self._svm_c(x, y, d, result_c, epoch)
        #ending
        self._w = w_vec[:-1]
        self._b = w_vec[-1]
        self._mins = mins
        self._maxs = maxs
        self._divisor = max(abs(w_vec.dot(r)) for r in x)
        #answers = [(1 if w_vec.dot(x[i]) > 0 else -1) for i in range(n)]
        #error_list = [(y[i], d[i]) for i in range(n) if y[i] != answers[i]]
        #err, w_err, fp_err, fn_err = self.get_all_errors(answers, error_list)
        self.simple_error = errors[index_min, 0]  #max(err, errors[index_min, 0])
        self.weighted_error = errors[index_min, 1]  #max(w_err, errors[index_min, 1])
        self.fp_error = errors[index_min, 2]  #max(fp_err, errors[index_min, 2])
        self.fn_error = errors[index_min, 3]  #max(fn_err, errors[index_min, 3])

    @staticmethod
    def get_all_errors(answers, error_list):
        simple_error = len(error_list)
        weighted_error = sum(t[1] for t in error_list)
        fp_error = sum(1 for t in error_list if t[0] == -1)
        fn_error = sum(1 for t in error_list if t[0] == 1)
        tlen = len(answers)
        tpos = sum(1 for x in answers if x == 1)
        tneg = tlen - tpos
        if tlen > 0:
            simple_error /= tlen
            weighted_error /= tlen
            if tpos > 0:
                fp_error /= tpos
            if tneg > 0:
                fn_error /= tneg
        return np.array([simple_error, weighted_error, fp_error, fn_error])

    def to_list(self):
        return [self._w, self._b, self._mins, self._maxs, self._divisor,
                [self.simple_error, self.weighted_error, self.fp_error, self.fn_error]]

    def from_list(self, list_):
        if len(list_) != 6:
            raise ValueError('from_list: len(list_) has to be 6')
        self._w, self._b, self._mins, self._maxs, self._divisor, errors = list_
        self.simple_error, self.weighted_error, self.fp_error, self.fn_error = errors

    def classify(self, x):
        return 1 if self.valuefy(x) > 0 else -1

    def classify_vec(self, vec, axis=-1):
        return np.apply_along_axis(self.classify, axis, vec)

    def valuefy(self, x):
        a = np.r_[x, x**2]
        return (self._w.dot((a - self._mins) / self._maxs) + self._b) / self._divisor

    def valuefy_vec(self, vec, axis=-1):
        return np.apply_along_axis(self.valuefy, axis, vec)

