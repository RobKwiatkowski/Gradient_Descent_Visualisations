import numpy as np


def batch_gradient_descent(X, y, learning_rate, a0_init=1.0, a1_init=1.0, i_max=250, tol=1e-5):

    if any(not isinstance(e, (int, float)) for e in [learning_rate, a0_init, i_max, tol]):
        print("Error! Only numerical inputs are allowed.")
        return False

    if type(a1_init) != np.float:
        a1_init = np.float(a1_init)

    if len(X) <= 2 or len(y) <= 2:
        print("Error! Dataset has to contain minimum 2 points!")
        return False

    if len(X) != len(y):
        print("Error! X and y have to be matching!")
        return False

    if learning_rate >= 1:
        print("Error! Learning rate cannot be bigger than 0! Algorithm will not converge!")
        return False

    X_b = np.c_[np.ones((len(X), 1)), X]

    pos = np.ndarray(shape=(2, 1), dtype=float, buffer=np.array([a0_init, a1_init]))
    steps_a0 = [pos[0][0]]
    steps_a1 = [pos[1][0]]

    m = len(X)
    i = 0

    for _ in range(i_max):
        gradients = 2/m * X_b.T.dot(X_b.dot(pos)-y)

        # early stopping:
        if np.all(np.abs(learning_rate * gradients) < tol):
            break
        else:
            pos = pos - learning_rate*gradients
            steps_a0.append(pos[0][0])
            steps_a1.append(pos[1][0])
            i += 1

    steps = np.column_stack((steps_a0, steps_a1))

    return [pos, steps, i]


rng = np.random.RandomState(1)
X = 5*rng.rand(100, 1)
y = 6 + 2*X + np.random.randn(100, 1)
