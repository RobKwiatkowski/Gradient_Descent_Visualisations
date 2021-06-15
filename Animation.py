import matplotlib.pyplot as plt
import numpy as np
import Batch_GD

rng = np.random.RandomState(1)
x = 5*rng.rand(100, 1)
y = 6 + 2*x + np.random.randn(100, 1)

results = Batch_GD.batch_gradient_descent(x, y, 0.05, 0, 0, 100, 0.001)

coefficients = results[0]
steps = results[1]
no_iterations = results[2]


def create_frames(x, y):
    i = 0
    files = []
    for s in steps:
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.xlabel("X", size=15)
        plt.xlim(0, 5)
        plt.xticks(size=15)

        plt.ylabel("y", size=15)
        plt.ylim(0, 20)
        plt.yticks(size=15)
        plt.grid()
        plt.scatter(x, y)

        tmp_x = np.array([0, 10])
        tmp_y = s[0] + s[1] * tmp_x
        plt.plot(tmp_x, tmp_y, c="r")
        plt.text(0.1, 18, "Iteration: {:03d}".format(i), size=20)
        plt.text(0.1, 16.5, "$a_0: {:.3f}$".format(s[0]), size=20)
        plt.text(0.1, 15.5, "$a_1: {:.3f}$".format(s[1]), size=20)

        frame = "Frames/{:03d}.png".format(i)
        files.append(frame)
        fig.savefig(frame, dpi=40)
        plt.close()
        i += 1

    print("All {} frames saved!".format(len(steps)))
