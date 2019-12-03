import cv2
import os
import numpy as np
import train.train_LRC

class Preprossesor :
    def __init__(self, comic_name):
        self.dir_path = os.path.join(os.path.dirname(__file__), comic_name)
        raw_data_path = os.path.join(self.dir_path, 'raw')
        path_list = os.listdir(raw_data_path)
        path_list.sort()
        self.path_list = [os.path.join(raw_data_path, x) for x in path_list]

        self.resized_data_path = os.path.join(self.dir_path, 'resized')
        if not os.path.exists(self.resized_data_path):
            os.makedirs(self.resized_data_path)

        self.edged_data_path = os.path.join(self.dir_path, 'edge')
        if not os.path.exists(self.edged_data_path):
            os.makedirs(self.edged_data_path)

        self.LRGT_data_path = os.path.join(self.dir_path, 'LRGT')
        if not os.path.exists(self.LRGT_data_path):
            os.makedirs(self.LRGT_data_path)


    def preprocess_save(self):
        for path in self.path_list:
            img = cv2.imread(path)
            ######################################################################################
            img_resized = cv2.resize(img[75:-75, :, :], (256, 256),interpolation = cv2.INTER_LINEAR)
            ######################################################################################
            ######################################################################################
            mask = np.mean(img, axis=2) < 80
            outline = (cv2.Canny(img, 100, 180) / 255).astype(np.bool)
            outline = np.logical_not(np.logical_or(outline, mask)).astype(np.uint8) * 255
            outline = cv2.resize(outline[75:-75, :], (256, 256),interpolation = cv2.INTER_AREA)
            ######################################################################################
            ######################################################################################
            img_LRGT = cv2.resize(img_resized, (32, 32), interpolation=cv2.INTER_LINEAR)
            img_LRGT = cv2.resize(img_LRGT, (256, 256), interpolation=cv2.INTER_LINEAR)
            ######################################################################################


            file_name = path[path.find('raw') + 5:]
            resized_path = os.path.join(self.resized_data_path, file_name)
            edged_path = os.path.join(self.edged_data_path, file_name)
            LRGT_path = os.path.join(self.LRGT_data_path, file_name)

            cv2.imwrite(resized_path, img_resized)
            cv2.imwrite(edged_path, outline)
            cv2.imwrite(LRGT_path, img_LRGT)


#
# P = Preprossesor('yumi_cell')
# P.preprocess_save()
#
# P = Preprossesor('more_data')
# P.preprocess_save()


# data_maker = train.train_LRC.CGAN(20, input_dir = "test") #반드시 LRC가 학습된 뒤에 실행
# data_maker.make_data_for_BD(2,2, start=0, end = 1000)
# data_maker = train.train_LRC.CGAN(20, input_dir = "yumi_cell") #반드시 LRC가 학습된 뒤에 실행
# data_maker.make_data_for_BD(2,2, start=0, end = 7380)
data_maker = train.train_LRC.CGAN(20, input_dir = "more_data") #반드시 LRC가 학습된 뒤에 실행
data_maker.make_data_for_BD(2,2, start=0, end = 5900)