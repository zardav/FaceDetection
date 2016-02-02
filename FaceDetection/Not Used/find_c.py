import numpy as np
from svm1 import svm
def find_c(c_arr, examples):
    X = examples[:,:-1]
    # constants
    cross_validation_times = 5
    learning_part = 0.7;
    n = examples.shape[0]
    middle = round(n*learning_part)
    # normalize
    mins = X.min(axis=0)
    X -= mins
    maxs = X.max(axis=0)
    X /= maxs
    c_len = c_arr.shape[0]
    errors = np.zeros(c_len)
    for j in range(c_len):
        c = c_arr[j]
        for _ in range(cross_validation_times):
            shuffled = np.random.permutation(examples)
            learnings, testings = np.split(shuffled, [middle])
            W = svm(learnings, c);
            error = sum([r[-1] * W.dot(r[:-1]) < 0 for r in testings])
        errors[j] += error/testings.shape[0]
    errors /= cross_validation_times
    result_c = c_arr[np.argmin(errors)]
    W = svm(examples, result_c);
    return (lambda x: W.dot((x - mins) / maxs)), errors
