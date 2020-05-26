import cv2
import os
import numpy as np
import util
import zipfile


class DataPreprocessor:
    def __init__(self):
        #################################################
        # Path setting
        self.input_path = '/coco/data_raw'
        self.output_resized_path = '/coco/data_resized_224x224'
        self.output_preprocessed_class_path = '/coco/data_preprocessed_PixColor_class' # 28 x 28 x 1 (class)
        self.output_preprocessed_ab_path = '/coco/data_preprocessed_PixColor_ab' # 28 x 28 x 2 (ab)

        if not (os.path.exists(self.input_path)):
            print("[unzip data]")
            os.mkdir(self.input_path)
            train_zip = zipfile.ZipFile('/coco/train2017.zip')
            train_zip.extractall(self.input_path)

        if not (os.path.exists(self.output_resized_path)):
            os.mkdir(self.output_resized_path)

        if not (os.path.exists(self.output_preprocessed_class_path)):
            os.mkdir(self.output_preprocessed_class_path)

        if not (os.path.exists(self.output_preprocessed_ab_path)):
            os.mkdir(self.output_preprocessed_ab_path)

        self.file_list_train = os.listdir(self.input_path)

        #################################################
        # Check whether gamut exist, if not, make gamut
        self.filename_gamut = str()
        list_filenames = os.listdir('./')

        for i in range(len(list_filenames)):
            if list_filenames[i][:5] == 'gamut':
                self.filename_gamut = list_filenames[i]
                break
            if i == len(list_filenames)-1:
                print('There is not gamut. Making gamut...')
                self.filename_gamut = util.save_gamut(10)
                print('Making gamut complete.')

        #################################################
        # Tablize gamut
        self.gamut = np.load(self.filename_gamut)
        self.num_colors = self.gamut.shape[0]
        self.table_gamut = util.tablize_gamut(self.gamut)

        #################################################
        # Make dictionary to count color(class) frequency
        # Use 263 bins, because I don't want to add 1 for every indices of frequency.
        # (Python index start from 0. but the color class start from 1.)
        # So, i just simply add a bin for class-0.(then the frequency of that bin must be always 0)
        self.dict_color_class_frequency = np.zeros((self.num_colors + 1,))


    def run(self):
        non_rgb_count = 0
        too_low_saturation_count = 0
        count = 0

        for i in self.file_list_train:
            img = cv2.imread(os.path.join(self.input_path, i))
            if len(img.shape) != 3:
                non_rgb_count += 1
                print('non colored one', non_rgb_count)
                continue

            saturation = np.mean((np.max(img, axis=2) - np.min(img, axis=2)) / (np.max(img, axis=2) + 1e-9))
            if saturation < 0.15:
                too_low_saturation_count += 1
                print('too low-saturated one', too_low_saturation_count)
                continue

            count += 1

            img_resized = self.resize_crop(img)
            # np.save(os.path.join(self.output_resized_path, i), img_resized)

            #################################################
            # Label's resolution must be 1/8 of input.
            # Because the network downsample resolution of input by 8.
            img_resized = cv2.resize(img_resized, (28, 28), interpolation = cv2.INTER_LINEAR)
            img_Lab = self.bgr2Lab(img_resized)
            np.save(os.path.join(self.output_preprocessed_ab_path, i), img_Lab[:, :, 1:])
            img_class = self.Lab2class(img_Lab)
            np.save(os.path.join(self.output_preprocessed_class_path, i), img_class)

            if (count % 1000) == 0:
                print("======%3.2f%%   [%7d/%7d]======" % (count / len(self.file_list_train) * 100, count, len(self.file_list_train)))

        print('%d image preprocessed. %d gray image dropped. %d low-saturated image dropped.' % (count, non_rgb_count, too_low_saturation_count))


    def resize_crop(self, img):
        h = img.shape[0]
        w = img.shape[1]

        if h > w:
            img = cv2.resize(img, (224, int(h / w * 224)), interpolation = cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (int(w / h * 224), 224), interpolation=cv2.INTER_LINEAR)

        h = img.shape[0]
        w = img.shape[1]

        return img[h//2 - 112:h//2 + 112, w//2 - 112:w//2 + 112, :]

    def bgr2Lab(self, img):
        img_Lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        return img_Lab

    def Lab2class(self, img):
        mapping_func_ab2class = np.vectorize(lambda a, b: self.table_gamut[a // 10, b // 10])
        img_class = mapping_func_ab2class(img[:, :, 1], img[:, :, 2])

        return img_class


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()
