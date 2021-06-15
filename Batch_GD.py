import numpy as np

rng = np.random.RandomState(1)
X = 5*rng.rand(100, 1)
y = 6 + 2*X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]


def batch_gradient_descent(X, learning_rate, a0_init=1.0, a1_init=1.0, iters=250, tol=1e-5):
    a = np.ndarray(shape=(2, 1), dtype=float, buffer=np.array([a0_init, a1_init]))
    steps_a0 = [a[0][0]]
    steps_a1 = [a[1][0]]

    m = len(X)
    i = 0

    for _ in range(iters):
        gradients = 2/m * X_b.T.dot(X_b.dot(a)-y)

        # early stopping:
        if np.any(np.abs(learning_rate * gradients)<tol):
            break
        else:
            a = a - learning_rate*gradients
            steps_a0.append(a[0][0])
            steps_a1.append(a[1][0])
            i += 1

    steps = np.column_stack((steps_a0,steps_a1))

    return a, steps, i


batch_gradient_descent(X, 0.1, 10.0, 0.0, 300, 0.001)
