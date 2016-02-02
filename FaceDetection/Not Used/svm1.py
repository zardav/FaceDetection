import numpy as np
from sub_gradient_loss import sub_gradient_loss


def svm(examples, c):
    numEx = examples.shape[0]
    alpha = 1
    EPOCH_TIMES = 3
    W = np.zeros(examples.shape[1]-1)
    t = 0
    for _ in range(EPOCH_TIMES):
        mixed_examples = np.random.permutation(examples)
        for r in mixed_examples:
            t += 1
            W -= alpha/t * sub_gradient_loss(r, W, c, numEx)
    return W
    
