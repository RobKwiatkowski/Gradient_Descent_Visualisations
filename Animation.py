import matplotlib.pyplot as plt
import numpy as np
import Batch_GD
import imageio
import os


def create_animation(x, y, gd_results, filename, frames_per_sec):

    """
    :param x: data points coordinates along x axis
    :param y: data points coordinates along y axis
    :param gd_results: results of gradient descent algorithm
    :param filename: desired name of .gif file; without extension
    :param frames_per_sec: desired fps of .gif
    :return: None; generates .gif file
    """

    if len(gd_results[1]) < 2:
        print("Error! Check steps history!")
        return False
    else:
        steps = results[1]

    if frames_per_sec < 0:
        print("Error! FPS has to be bigger than 0.")
        return False

    if not isinstance(filename, str):
        print("Error! Check filename!")
        return False

    # creating frames
    i = 0
    frames = []
    dir_name = "tmp"

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " created. ")
    else:
        print("Directory ", dir_name, " already exists.")

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

        frame = "tmp/{:03d}.png".format(i)
        frames.append(frame)
        fig.savefig(frame, dpi=40)
        plt.close()
        i += 1

    print("All {} frames saved!".format(len(steps)))

    # creating animation
    with imageio.get_writer(filename+".gif", mode="I", fps=frames_per_sec) as writer:
        for name in frames:
            im = imageio.imread(name)
            writer.append_data(im)

    # cleaning up
    for item in frames:
        if item.endswith(".png"):
            os.remove(item)
    print(".gif successfully created!")


rng = np.random.RandomState(1)
x_sample = 5*rng.rand(100, 1)
y_sample = 6 + 2*x_sample + np.random.randn(100, 1)

results = Batch_GD.batch_gradient_descent(x_sample, y_sample, 0.05, 0, 0, 100, 0.001)

create_animation(x_sample, y_sample, results, "test", 2)
