"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 1015 1539, last modified in 2020 1015 1555.
"""

import numpy as np


class Clustering:
    """
    The origin class of all clustering algorithms.
    :param
        para_dis:
            The distances matrix.
        para_idx:
            The index of clustering data.
    @attribute
        centers:
            The clustering centers.
        blocks:
            The clustered blocks.
        lab:
            The instances label, and the instance lab will be same as the index of block.
    """

    def __init__(self, para_dis, para_idx=None):
        """
        The constructor.
        """
        self.dis = para_dis
        self.idx = para_idx
        self.centers = []
        self.blocks = []
        self.lab = []
        self.max_dis = 0
        self.ave_dis = 0
        self.num_sample = 0
        self.__initialize()

    def __initialize(self):
        """
        The initialize of Clustering.
        """

        if self.idx is None:
            self.idx = np.arange(len(self.dis))

        self.max_dis = np.max(self.dis[self.idx, :][:, self.idx])
        self.num_sample = len(self.idx)
        self.ave_dis = np.sum(np.triu(self.dis[self.idx, :][:, self.idx])) / (self.num_sample * (
                self.num_sample - 1) / 2) if self.num_sample > 1 else np.sum(self.dis)
