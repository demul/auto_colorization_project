import numpy as np
import cv2
import os
import time

class Loader:
    def __init__(self, is_test=False, dir = "yumi_cell", train_size = 7000, test_size = 300):
        np.random.seed(0)

        self.dir = dir
        ### 뒤 380컷은 테스트용으로 쓰자
        idx = np.arange(0, train_size + test_size)
        if (is_test) :
            self.idx = idx[train_size:]  # 300
        else :
            self.idx = idx[:train_size] # 7000
            np.random.shuffle(self.idx)


        self.cursor = 0

    def shuffle (self):
        np.random.shuffle(self.idx)


    def next(self, batch_size):
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

        idx_list = self.idx[self.cursor : self.cursor + batch_size]
        self.cursor += batch_size
        if self.cursor + batch_size > len(self.idx) :
            self.cursor = 0

        for i, idx_num in enumerate(idx_list):
            # load img
            img_edge = cv2.imread(os.path.join(self.dir, "edge", "%4d.jpg"%idx_num), cv2.IMREAD_GRAYSCALE)
            img_GT = cv2.imread(os.path.join(self.dir, "resized", "%4d.jpg"%idx_num))

            edge_batch[i] = np.expand_dims(img_edge, axis=3)
            GT_batch[i] = img_GT

        return edge_batch, GT_batch


    def next_LRGT(self, batch_size):
        LRGT_batch = np.empty((
            batch_size,
            256,
            256,
            3
        ), dtype=np.float32)

        idx_list = self.idx[self.cursor: self.cursor + batch_size]
        #### next "앞"에 붙어서 사용될 것이므로 주석처리
        # self.cursor += batch_size
        # if self.cursor + batch_size >= len(self.idx):
        #     self.cursor = 0
        #############################################
        for i, idx_num in enumerate(idx_list):
            # load imgimpo
            img_LRGT = cv2.imread(os.path.join(self.dir, "LRGT", "%4d.jpg" % idx_num), cv2.IMREAD_COLOR)
            LRGT_batch[i] = img_LRGT

        return LRGT_batch


    def next_LRC(self, batch_size):
        LRC_batch = np.empty((
            batch_size,
            256,
            256,
            3
        ), dtype=np.float32)

        idx_list = self.idx[self.cursor: self.cursor + batch_size]
        #### next "앞"에 붙어서 사용될 것이므로 주석처리
        # self.cursor += batch_size
        # if self.cursor + batch_size >= len(self.idx):
        #     self.cursor = 0
        #############################################
        for i, idx_num in enumerate(idx_list):
            # load imgimpo
            img_LRC = cv2.imread(os.path.join(self.dir, "LRC", "%4d.png"%idx_num), cv2.IMREAD_COLOR)
            LRC_batch[i] = img_LRC

        return LRC_batch