import numpy as np
import matplotlib.pyplot as plt


def random_rectangle(type: str = "mono", **kwargs):
    """Random rectangle function wrapper"""
    assert type in ["mono", "rgb"]
    if type == "mono":
        return random_rectangle_single_channel(**kwargs)

    elif type == "rgb":
        return random_rectangle_rbg(**kwargs)


def random_rectangle_single_channel(full_size: int = 16,
                                    seed: int = 1234):
    """Random rectangle in single channel"""
    # np.random.seed(seed)

    max_rectangle_size = full_size // 2
    w, h = np.random.randint(1, max_rectangle_size, size=2)

    # lower left corner of the rectangle
    bx, by = np.random.randint(0, full_size - 1 - w), np.random.randint(0, full_size - 1 - h)

    img = np.zeros((full_size, full_size), dtype=float)
    img[by:by + h, bx:bx + w] = 1.0

    return img, (bx, by, w, h)


def random_rectangle_rbg(full_size: int = 16,
                         seed: int = 1234, ):
    """Random rectangle in rbg channel"""
    # np.random.seed(seed)
    random_channel = np.random.randint(0, 2)

    img = np.zeros((3, full_size, full_size))
    img[random_channel], (bx, by, w, h) = random_rectangle_single_channel(full_size, seed)

    return img, (bx, by, w, h, random_channel)


def plot_helper(img, ax=None):
    if ax == None:
        fig, ax = plt.subplots()
    sizes = img.shape
    if len(sizes) == 2:
        ax.imshow(img, origin='lower', cmap='binary')

    elif len(sizes) == 3:
        # img = - img + 1 # plot zeros as white, ones as color
        # img[img==1.0] = 0.0
        # cmap = plt.cm.gray
        # cmap.set_bad(color='white')
        ax.imshow(img.transpose([1, 2, 0]))

    ax.set_xticks([])
    ax.set_yticks([])


def plot_result(img, data, ax=None):
    if ax == None:
        fig, ax = plt.subplots()

    if len(img.shape) == 2:
        plot_helper(img, ax)
        bx, by, w, h = data
        xs = np.array([bx - 0.5, bx + w - 0.5, bx + w - 0.5, bx - 0.5, bx - 0.5])
        ys = np.array([by - 0.5, by - 0.5, by + h - 0.5, by + h - 0.5, by - 0.5])
        ax.plot(xs, ys, lw=2, c='red')


    if len(img.shape) == 3:
        plot_helper(img, ax)
        bx, by, w, h, _ = data
        xs = np.array([bx - 0.5, bx + w - 0.5, bx + w - 0.5, bx - 0.5, bx - 0.5])
        ys = np.array([by - 0.5, by - 0.5, by + h - 0.5, by + h - 0.5, by - 0.5])
        ax.plot(xs, ys, lw=2, c='white')
