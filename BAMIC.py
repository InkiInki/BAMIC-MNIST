"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 1123, last modified in 2020 1206.
"""

import numpy as np
from Prototype import MIL
from kMeans import KMeans
from B2B import B2B
from FunctionTool import get_k_cross_validation_index


class BaMic(MIL):
    """
    @param:
        Please refer miVLAD.
    """

    def __init__(self, path, k=10, k_m=0.99, b2b='ave_hausdorff', bag_space=None):
        """
        The constructor.
        """
        super(BaMic, self).__init__(path, bag_space=bag_space)
        self.k = k
        self.k_m = k_m
        self.b2b = b2b
        self.dis = []
        self.tr_idx = []
        self.te_idx = []
        self.__initialize_bamic()

    def __initialize_bamic(self):
        """
        The initialize of BAMIC.
        """
        self.dis = B2B(self.data_name, self.bag_space, self.num_att, self.b2b).get_dis()

    def get_mapping(self):
        """
        Get mapping.
        """
        self.tr_idx, self.te_idx = get_k_cross_validation_index(self.num_bag)
        for loop_k in range(self.k):
            temp_tr_idx = self.tr_idx[loop_k]
            temp_centers = KMeans(self.dis, np.array(temp_tr_idx),
                                  int(self.k_m * min(100, len(temp_tr_idx)))).clustering()
            temp_dis = self.dis[:, temp_centers]
            yield temp_dis[temp_tr_idx], self.bag_lab[temp_tr_idx], \
                  temp_dis[self.te_idx[loop_k]], self.bag_lab[self.te_idx[loop_k]], None
