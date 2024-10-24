import torch
import random

class FeatureShuffling():
    def __init__(self):
        super(FeatureShuffling, self).__init__()

    def __call__(self, features1, features2):
        size, channel = features1.size(0), features1.size(1)
        num = 4

        a = list(range(size))
        b = [a[i: i + num] for i in range(0, size, num)]
        c = []
        for b1 in b:
            random.shuffle(b1)
            c.extend(b1)

        shuffling_features1 = torch.zeros([size, channel]).cuda()
        shuffling_features2 = torch.zeros([size, channel]).cuda()

        for i in range(size):
            shuffling_features1[i] = features1[c[i]]
            shuffling_features2[i] = features2[c[i]]
        return shuffling_features1, shuffling_features2


