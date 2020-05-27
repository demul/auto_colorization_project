import numpy as np
import cv2
# import matplotlib.pyplot as plt


def save_gamut(quantization_grid):
    #################################################
    # I get CIE-Lab space from uint8-type RGB space
    # Because opencv transform float32-type RGB space to CIE-Lab space in range :
    # 0.0 <= L <= 100.0
    # -86.1813 <= a <= 98.2352
    # -107.862 <= b <= 94.4758
    #
    # The negative values of a, b make gamut-tablizing(tablize_gamut()) complex.
    # Because index of np.array must be positive or 0
    # So, i transform uint8-type RGB space to CIE-Lab space in range :
    # 0 <= L <= 255
    # 42 <= a <= 226
    # 20 <= b <= 223
    #
    # For simplifying gamut-tablizing

    #################################################
    # My quantization method is simple.
    # Floor-divide values of a, b with quantization_grid.
    # And multiply it again with quantization_grid.
    # Quantization_grid is hyperparameter.
    # I only experiment the grid-setting on 10
    # A total of 262 a, b pairs are in gamut.
    set_pairs = set()
    for red in range(256):
        for green in range(256):
            for blue in range(256):
                im = np.array((blue, green, red), np.uint8).reshape(1, 1, 3)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

                ## quantize
                a = (im[0, 0, 1] // quantization_grid) * quantization_grid
                b = (im[0, 0, 2] // quantization_grid) * quantization_grid

                set_pairs.add((a, b))

    set_pairs = list(set(set_pairs))  # deduplication
    filename_gamut = 'gamut_grid%d_class%d_uint8.npy' % (quantization_grid, len(set_pairs))
    np.save(filename_gamut, np.array(list(set_pairs)))
    ## visualization
    # plt.scatter([x[0] for x in gamut_quantized], [x[1] for x in gamut_quantized])
    # plt.xlabel('a')
    # plt.ylabel('b')
    # plt.show()

    return filename_gamut


def tablize_gamut(gamut):
    #################################################
    # The index of this table is values of a, b divided by 10
    gamut = gamut // 10
    len_gamut = gamut.shape[0]
    max_ab = np.max(gamut, axis=0)

    #################################################
    ## max_ab + 2 = max_ab + 1 + 1
    ## + 1 (because max index of np.array is width - 1(or height - 1))
    ## + 1 (for padding right, bottom side of the table by 1 for simplify soft encoding)
    table_gamut = np.zeros([max_ab[0] + 2, max_ab[1] + 2], dtype=np.uint32)

    for i in range(len_gamut):
        #################################################
        ## Value of index start from 1,
        ## Because 0 is a value for index-pairs which is not in gamut.
        table_gamut[tuple(gamut[i])] = i + 1

    return table_gamut


def Lab2bgr(batch_L, batch_ab):
    # batch_L : [None, 224, 224, 1]
    # batch_ab : [None, 56, 56, 2]
    # batch_bgr : [None, 224, 224, 3]
    batch_size = batch_L.shape[0]
    h = batch_L.shape[1]
    w = batch_L.shape[2]

    batch_bgr = np.empty([batch_size, h, w, 3], dtype=np.uint8)

    for idx in range(len(batch_ab)):
        # resize batch_ab
        ab_resized = cv2.resize(batch_ab[idx], (224, 224), interpolation=cv2.INTER_LINEAR)
        # concat Lab
        Lab = np.concatenate([batch_L[idx], ab_resized], axis=2).astype(np.uint8)
        # convert Lab to bgr
        batch_bgr[idx] = cv2.cvtColor(Lab, cv2.COLOR_LAB2BGR)

    return batch_bgr


class Decoder:
    def __init__(self, table_gamut, temperature=0.38, sampling='probabilistic'):
        self.table_gamut = table_gamut
        self.table_gamut_inverse = self.get_inverse_table(table_gamut)
        self.temperature = temperature
        if sampling == 'probabilistic':
            self.decode = self.probabilistic_decode
        else:
            self.decode = self.annealed_mean_decode


    def softmax_temperature_scaled(self, prob):
        #################################################
        # Reformulation to avoid log(0)
        #   exp(log(probability) / T) / sum(exp(log(probability) / T))
        # = probability^(1/T) / sum(probability^(1/T))
        prob_scaled = (prob ** (1 / self.temperature)) / np.sum(prob ** (1 / self.temperature))
        return prob_scaled


    def annealed_mean_decode(self, encoding):
        # softencoding : [262,]
        encoding_scaled = self.softmax_temperature_scaled(encoding)
        annealed_mean = np.zeros([2, ], dtype=np.float32)

        for idx, prob in enumerate(encoding_scaled):
            ab = self.table_gamut_inverse[idx + 1] * 10
            annealed_mean += ab * prob

        return annealed_mean.astype(np.uint8)

    def probabilistic_decode(self, encoding):
        # encoding : [262,]
        # class_sampled = np.random.choice(262, 1, p=self.softmax_temperature_scaled(encoding))[0]
        class_sampled = np.random.choice(262, 1, p=encoding)[0]
        # class_sampled = np.argmax(encoding)
        ab = self.table_gamut_inverse[class_sampled + 1] * 10

        return ab.astype(np.uint8)


    def get_inverse_table(self, table_gamut):
        table_gamut_inverse = dict()

        for i in range(1, table_gamut.max()+ 1):
            a = np.where(table_gamut == i)[0][0]
            b = np.where(table_gamut == i)[1][0]
            table_gamut_inverse[i] = np.array([a, b], dtype=np.uint8)

        return table_gamut_inverse


    def encoding2ab(self, batch_encoding):
        # batch_encoding : [None, 262]
        batch_size = batch_encoding.shape[0]

        batch_ab = np.empty([batch_size, 2], dtype=np.uint8)

        for idx, encoding in enumerate(batch_encoding):
            batch_ab[idx] = self.decode(encoding)
        return batch_ab
