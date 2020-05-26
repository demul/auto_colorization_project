import numpy as np
import os
import util


class DataLoader:
    def __init__(self, validation_len=1000):
        self.input_path = '/coco/data_resized_224x224'
        self.label_class_path = '/coco/data_preprocessed_PixColor_class'
        self.label_ab_path = '/coco/data_preprocessed_PixColor_ab'

        self.train_path_list = [os.path.join(self.input_path, x) for x in os.listdir(self.input_path)]  # 103,951
        self.val_path_list = self.train_path_list[-validation_len:]  # 1000
        self.train_path_list = self.train_path_list[:-validation_len]  # 102,951

        self.train_label_class_path_list = [os.path.join(self.label_class_path, x) for x in os.listdir(self.label_class_path)]  # 103,951
        self.val_label_class_path_list = self.train_label_class_path_list[-validation_len:]  # 1000
        self.train_label_class_path_list = self.train_label_class_path_list[:-validation_len]  # 102,951

        self.train_label_ab_path_list = [os.path.join(self.label_ab_path, x) for x in os.listdir(self.label_ab_path)]  # 103,951
        self.val_label_ab_path_list = self.train_label_ab_path_list[-validation_len:]  # 1000
        self.train_label_ab_path_list = self.train_label_ab_path_list[:-validation_len]  # 102,951


        self.idx_train = [i for i in range(len(self.train_path_list))]
        np.random.shuffle(self.idx_train)
        self.idx_val = [i for i in range(validation_len)]

        self.cursor_train = 0
        self.cursor_val = 0

        #################################################
        # check whether gamut exist, if not, make gamut
        self.filename_gamut = str()
        list_filenames = os.listdir('./')

        for i in range(len(list_filenames)):
            if list_filenames[i][:5] == 'gamut':
                self.filename_gamut = list_filenames[i]
                break
            if i == len(list_filenames)-1:
                print('There is not gamut. Run \'data_preprocessor.py\' before run this script...')

        #################################################
        # tablize gamut
        self.gamut = np.load(self.filename_gamut)
        self.num_colors = self.gamut.shape[0]
        self.table_gamut = util.tablize_gamut(self.gamut)


    def next_train(self, batch_size):
        batch_img = np.empty((
            batch_size,
            224,
            224,
            1
        ), dtype=np.float32)

        batch_label_encoding = np.empty((
            batch_size,
            28,
            28,
            262
        ), dtype=np.float32)

        batch_label_ab = np.empty((
            batch_size,
            28,
            28,
            2
        ), dtype=np.float32)

        for idx, val in enumerate(self.idx_train[self.cursor_train:self.cursor_train + batch_size]):
            filename_input = self.train_path_list[val]
            filename_label_class = self.train_label_class_path_list[val]
            filename_label_ab = self.train_label_ab_path_list[val]

            batch_img[idx] = np.expand_dims(np.mean(np.load(filename_input), axis=2), axis=2)
            # [None, 224, 224, 1]
            batch_label_encoding[idx] = self.class2onthotencoding(np.load(filename_label_class))
            # [None, 28, 28, 262]
            batch_label_ab[idx] = np.load(filename_label_ab)
            # [None, 28, 28, 2]

        self.cursor_train += batch_size
        if self.cursor_train + batch_size > len(self.idx_train):
            self.cursor_train = 0
            np.random.shuffle(self.idx_train)

        return batch_img, batch_label_encoding, batch_label_ab


    def next_val(self, batch_size):
        batch_GT = np.empty((
            batch_size,
            224,
            224,
            3
        ), dtype=np.float32)

        batch_img = np.empty((
            batch_size,
            224,
            224,
            1
        ), dtype=np.float32)

        batch_label_class = np.empty((
            batch_size,
            28,
            28,
            262
        ), dtype=np.float32)

        batch_label_ab = np.empty((
            batch_size,
            28,
            28,
            2
        ), dtype=np.float32)


        for idx, val in enumerate(self.idx_val[self.cursor_val:self.cursor_val + batch_size]):
            filename_input = self.val_path_list[val]
            filename_label_class = self.val_label_class_path_list[val]
            filename_label_ab = self.val_label_ab_path_list[val]

            batch_GT[idx] = np.load(filename_input)
            # [None, 224, 224, 3]
            batch_img[idx] = np.expand_dims(np.mean(batch_GT[idx], axis=2), axis=2)
            # [None, 224, 224, 1]
            batch_label_class[idx] = self.class2onthotencoding(np.load(filename_label_class))
            # [None, 28, 28, 262]
            batch_label_ab[idx] = np.load(filename_label_ab)
            # [None, 28, 28, 2]

        self.cursor_val += batch_size
        if self.cursor_val + batch_size > len(self.idx_val):
            self.cursor_val = 0

        return batch_img, batch_label_class, batch_label_ab, batch_GT


    def class2onthotencoding(self, img_class):
        #################################################
        # Encoding's dimension is (height, width, number of classes)
        h = img_class.shape[0]
        w = img_class.shape[1]

        # Transform numeric class to one-hot-encoding
        encoding = np.eye(self.num_colors)[img_class - 1] # 1 ~ 262 => 0 ~ 261 (to fit on index number)

        return encoding


if __name__ == "__main__":
    DL = DataLoader()
    img, lb_class, lb_ab = DL.next_train(100)

