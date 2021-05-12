"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 1015, last modified in 2020 1221.
@note: Only accept distance matrix.
"""

import numpy as np

from ClusteringOrigin import Clustering


class KMeans(Clustering):

    """
    The density peaks clustering algorithm.
    :param
        k:
            The number of clustering centers.
    @attribute
        centers, lab, blocks:
            Please refer the class named ClusteringOrigin.
    @example

    """

    def __init__(self, dis, idx=None, k=10, max_iter=10):
        """
        The constructor.
        """
        super(KMeans, self).__init__(dis, idx)
        self.dis = dis
        self.ids = idx
        self.k = k
        self.max_iter = max_iter
        self.__initialize_kmeans()

    def __initialize_kmeans(self):
        """
        The initialize of density peaks.
        """
        if self.idx is None:
            self.idx = list(range(len(self.dis)))

    def __is_centers_equal(self, last_centers, current_centers):
        """
        If last centers equal current centers, return True else False.
        """
        last_sorted_centers = np.sort(last_centers)
        current_sorted_centers = np.sort(current_centers)
        for i in range(self.k):
            if last_sorted_centers[i] != current_sorted_centers[i]:
                return False
        return True

    def clustering(self):
        """
        Clustering to k blocks.
        """

        # Step 1. Initialize the centers with random generate.
        temp_num_sample = len(self.idx)
        temp_random_idx = np.random.permutation(temp_num_sample)
        temp_last_center = temp_random_idx[-self.k:]
        self.centers = temp_random_idx[:  self.k]

        loop = 0
        while self.__is_centers_equal(temp_last_center, self.centers) is not True and loop < self.max_iter:
            temp_last_center = np.copy(self.centers)
            temp_dis = self.dis[self.idx][:, temp_last_center]
            self.blocks = [[center] for center in self.centers]
            self.lab = np.zeros(temp_num_sample)
            self.lab[self.centers] = list(range(self.k))
            for idx_sample in range(temp_num_sample):
                if idx_sample in self.centers:
                    continue
                temp_sorted_dis_idx = np.argsort(temp_dis[idx_sample])
                self.lab[idx_sample] = temp_sorted_dis_idx[0]
                self.blocks[temp_sorted_dis_idx[0]].append(idx_sample)

            for idx_k in range(self.k):
                temp_dis = self.dis[self.blocks[idx_k]][:, self.blocks[idx_k]]
                temp_sum_dis = np.sum(temp_dis, 0)
                temp_sorted_dis_idx = np.argsort(temp_sum_dis)
                self.centers[idx_k] = self.blocks[idx_k][temp_sorted_dis_idx[0]]
            loop += 1
        return self.idx[self.centers]


if __name__ == '__main__':
    # np.random.seed(10)
    d1 = np.random.randn(100, 100)
    d1 = np.triu(d1)
    d1 = np.transpose(d1) + d1
    dp = KMeans(d1, k=2, max_iter=100)
    dp.clustering()
