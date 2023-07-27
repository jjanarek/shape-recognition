import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SimpleModel(nn.Module):
    def __init__(self, xy_size, hidden_1=128, hidden_2=256):
        super(SimpleModel, self).__init__()
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.xy_size = xy_size

        self.fc1 = nn.Linear(self.xy_size, self.hidden_1)
        self.fc2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(self.hidden_2, 4)

    def forward(self, x):
        x = x.view(-1, self.xy_size)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))

        return x


# functions for evaluation of the model
def intersection_over_union(bbox0: torch.Tensor, bbox1: torch.Tensor) -> float:
    bbox0, bbox1 = bbox0.detach().numpy().flatten(), bbox1.detach().numpy().flatten()

    # print(bbox0, bbox1)

    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    intersection_x = np.max((0, np.min((x0 + w0, x1 + w1)))) - np.max((0, np.max((x0, x1))))
    intersection_y = np.max((0, np.min((y0 + h0, y1 + h1)))) - np.max((0, np.max((y0, y1))))

    intersection = intersection_x * intersection_y
    union = h0 * w0 + h1 * w1 - intersection

    return intersection / union


def plot_bbox(bbox, ax, color='C0', max_size=8):
    bbox = bbox.detach().numpy().flatten()
    x, y, w, h = bbox
    img = np.zeros((max_size, max_size))
    ax.imshow(img, cmap='gray', alpha=0.0)
    ax.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], color=color)


def plot_performance(model, examples, targets):

    n_examples = len(examples)
    fig, axs = plt.subplots(ncols=n_examples, figsize=(20,20))

    for i, ax in enumerate(axs):

        bbox0 = targets[i]
        bbox1 = model(examples[i])
        plot_size = examples[i].squeeze().size()[0]

        ax.set_xticks(np.arange(0, plot_size, 1))
        ax.set_yticks(np.arange(0, plot_size, 1))

        ax.grid()
        plot_bbox(bbox0, ax, max_size=plot_size)
        plot_bbox(bbox1, ax, 'red', max_size=plot_size)

        iou = intersection_over_union(bbox0, bbox1)
        x, y, w, h = bbox0 # target bbox
        ax.text(x, y - 0.15, "IoU={:.2f}".format(iou), fontsize=10)

    return fig

