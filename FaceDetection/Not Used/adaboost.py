import numpy as np


def ada_boost(self, classifiers):
    _T = len(classifiers)  # Num of classifiers
    eps_vec = np.empty(_T)  # Size of error for each classifier
    alpha_vec = np.empty(_T)  # Reliability for each classifier
    m = self.num_examples  # m is amount of examples
    _D = np.ones(m) / m  # Distribution probability for each example
    epoch = 4
    for _ in range(epoch):
        for t in range(_T):
            #  Weights are made by the distribution probability of the example (in _D vector, as mentioned above)
            h = classifiers[t]  # h is the current classifier
            h.learn(d=_D)
            # Applying the classify on all of our training data, and save results vector (results in {1,-1})
            classifying_results = h.svm.classify_vec(h.examples[:, :-1])
            error_list = classifying_results != h.examples[:, -1]
            # Summing errors when each error multiplied in its probability
            eps_vec[t] = (_D * error_list).sum()
            alpha_vec[t] = np.log((1 - eps_vec[t]) / eps_vec[t]) / 2  # Formula of alpha
            # Formula to change probability for each example
            _D *= np.exp(-alpha_vec[t]*h.examples[:, -1]*classifying_results)
            _D /= _D.sum()  # Normalization distribution probability
    return alpha_vec
