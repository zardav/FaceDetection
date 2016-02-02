def sub_gradient_loss(example, W, c, n):
    (x, y) = example[:-1], example[-1]
    grad_loss = W / n
    if 1 - y * W.dot(x) > 0:
        grad_loss -= c*y * x
    return grad_loss
