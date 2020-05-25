import numpy as np
import os
import util


class DataLoader:
    def __init__(self, validation_len=1000, softencoding_variance=5):
        self.input_path = '/coco/data_resized_56x56'
        self.label_path = '/coco/data_preprocessed_CIC'

        self.train_path_list = [os.path.join(self.input_path, x) for x in os.listdir(self.input_path)]  # 103,951
        self.val_path_list = self.train_path_list[-validation_len:]  # 1000
        self.train_path_list = self.train_path_list[:-validation_len]  # 102,951

        self.train_label_path_list = [os.path.join(self.label_path, x) for x in os.listdir(self.label_path)]  # 103,951
        self.val_label_path_list = self.train_label_path_list[-validation_len:]  # 1000
        self.train_label_path_list = self.train_label_path_list[:-validation_len]  # 102,951

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

        #################################################
        # get 'class to soft-encoding' mapping table
        self.table_class2softencoding = util.get_softencoding_mapping(self.num_colors, self.table_gamut, softencoding_variance)


    def next_train(self, batch_size):
        batch_img = np.empty((
            batch_size,
            224,
            224,
            1
        ), dtype=np.float32)

        batch_label = np.empty((
            batch_size,
            56,
            56,
            262
        ), dtype=np.float32)

        for idx, val in enumerate(self.idx_train[self.cursor_train:self.cursor_train + batch_size]):
            filename_input = self.train_path_list[val]
            filename_label = self.train_label_path_list[val]

            batch_img[idx] = np.expand_dims(np.mean(np.load(filename_input), axis=2), axis=2)
            batch_label[idx] = self.class2softendoing(np.load(filename_label))

        self.cursor_train += batch_size
        if self.cursor_train + batch_size > len(self.idx_train):
            self.cursor_train = 0
            np.random.shuffle(self.idx_train)

        return batch_img, batch_label


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

        batch_label = np.empty((
            batch_size,
            56,
            56,
            262
        ), dtype=np.float32)


        for idx, val in enumerate(self.idx_val[self.cursor_val:self.cursor_val + batch_size]):
            filename_input = self.val_path_list[val]
            filename_label = self.val_label_path_list[val]

            batch_GT[idx] = np.load(filename_input)
            batch_img[idx] = np.expand_dims(np.mean(batch_GT[idx], axis=2), axis=2)
            # [None, 224, 224, 1]
            batch_label[idx] = self.class2softendoing(np.load(filename_label))
            # [None, 56, 56, 262]

        self.cursor_val += batch_size
        if self.cursor_val + batch_size > len(self.idx_val):
            self.cursor_val = 0

        return batch_img, batch_label, batch_GT


    def class2softendoing(self, img_class):
        #################################################
        # Encoding's dimension is (height, width, number of classes)
        h = img_class.shape[0]
        w = img_class.shape[1]
        encoding = np.empty([h, w, self.num_colors])

        # Transform numeric class to soft-encoding
        for y in range(h):
            for x in range(w):
                encoding[y, x] = self.table_class2softencoding[img_class[y, x]]

        return encoding


# if __name__ == "__main__":
#     DL = DataLoader()
#     img, lb = DL.next_train(100)