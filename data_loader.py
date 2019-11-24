import numpy as np
import cv2
import os
import time

class Loader:
    def __init__(self):
        np.random.seed(int(time.time()))

        ### 뒤 380컷은 테스트용으로 쓰자
        idx = np.arange(0, 7380)
        self.train_idx = idx[:7380 - 380] # 7000
        self.test_idx = idx[7380 - 380:] # 380

        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.test_idx)

        self.cursor = 0

    def shuffle (self):
        np.random.shuffle(self.train_idx)

    def next(self, batch_size, test=False):
        edge_batch = np.empty((
            batch_size,
            256,
            256,
            1
        ), dtype=np.float32)

        GT_batch = np.empty((
            batch_size,
            256,
            256,
            3
        ), dtype=np.float32)

        if test:
            idx_list = self.test_idx
        else:
            idx_list = self.train_idx

        idx_list = idx_list[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size
        if self.cursor >= 7000 :
            self.cursor = 0

        for i, idx_num in enumerate(idx_list):
            # load img
            img_edge = cv2.imread(os.path.join("yumi_cell", "edge", "%4d.jpg"%idx_num), cv2.IMREAD_GRAYSCALE)
            img_GT = cv2.imread(os.path.join("yumi_cell", "resized", "%4d.jpg"%idx_num))

            edge_batch[i] = np.expand_dims(img_edge, axis=3)
            GT_batch[i] = img_GT

        return edge_batch, GT_batch

    def next_LRC(self, batch_size, test=False):
        LRC_batch = np.empty((
            batch_size,
            256,
            256,
            3
        ), dtype=np.float32)

        if test:
            idx_list = self.test_idx
        else:
            idx_list = self.train_idx

        idx_list = idx_list[self.cursor: self.cursor + batch_size]
        #### next "앞"에 붙어서 사용될 것이므로 주석처리
        # self.cursor += batch_size
        # if self.cursor >= 7000:
        #     self.cursor = 0
        #############################################
        for i, idx_num in enumerate(idx_list):
            # load imgimpo
            img_LRC = cv2.imread(os.path.join("yumi_cell", "LRC", "%4d.png"%idx_num), cv2.IMREAD_COLOR)
            LRC_batch[i] = img_LRC

        return LRC_batch