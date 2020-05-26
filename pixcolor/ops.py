import tensorflow as tf
import numpy as np


def conv(x, w, isTrain,
         conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=False, activation=tf.nn.relu):
    if conv_dilation == 1:
        res = tf.nn.conv2d(x, w, strides=conv_stride, padding=conv_padding)
    else:
        res = tf.nn.atrous_conv2d(x, w, conv_dilation, padding=conv_padding)

    if batch_norm is True:
        res = tf.layers.batch_normalization(res, training=isTrain)

    if activation is not None:
        res = activation(res)

    return res


def masked_conv_typeA(x, w, isTrain, wd,
         conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=False, activation=tf.nn.relu):
    ksize = w.shape[0].value
    ch_input = w.shape[2].value
    ch_output = w.shape[3].value

    mask = np.ones([ksize, ksize, ch_input, ch_output])
    mask[ksize // 2, ksize // 2:] = 0
    mask[ksize // 2 + 1:, :] = 0
    tensor_mask = tf.constant(mask, dtype=tf.float32)
    w_masked = w * tensor_mask
    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(w_masked), wd))

    res = conv(x, w_masked, isTrain,
         conv_stride=conv_stride, conv_padding=conv_padding, conv_dilation=conv_dilation, batch_norm=batch_norm, activation=activation)

    return res


def masked_conv_typeB(x, w, isTrain, wd,
         conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=False, activation=tf.nn.relu):
    ksize = w.shape[0].value
    kch_input = w.shape[2].value
    kch_output = w.shape[3].value

    mask = np.ones([ksize, ksize, kch_input, kch_output])
    mask[ksize // 2, ksize // 2 + 1:] = 0
    mask[ksize // 2 + 1:, :] = 0
    tensor_mask = tf.constant(mask, dtype=tf.float32)
    w_masked = w * tensor_mask
    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(w_masked), wd))

    res = conv(x, w_masked, isTrain,
         conv_stride=conv_stride, conv_padding=conv_padding, conv_dilation=conv_dilation, batch_norm=batch_norm, activation=activation)

    return res

def split_into_stacks(x):
    # [None, 28, 28, 64]
    paddings = tf.constant([[0, 0], [1, 0], [0, 0], [0, 0]])  # pad top of the input by 1
    x_vertical = tf.pad(x, paddings, 'CONSTANT') # [None, 29, 28, 64]
    x_horizontal = x # [None, 28, 28, 64]
    return x_vertical, x_horizontal


class gated_conv_unit:
    def __init__(self, name, wd, ch=64, ksize=5):
        W_vertical = tf.get_variable(name + '_vertical', shape=[5, 5, ch, ch * 2],
                                    dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        mask_vertical = np.ones([ksize, ksize, ch, ch * 2])
        mask_vertical[ksize // 2 + 1:, :] = 0
        tensor_mask_vertical = tf.constant(mask_vertical, dtype=tf.float32)
        self.W_vertical = W_vertical * tensor_mask_vertical
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_vertical), wd))

        W_horizontal = tf.get_variable(name + '_horizontal', shape=[1, 5, ch, ch * 2],
                                          dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        mask_horizontal = np.ones([1, ksize, ch, ch * 2])
        mask_horizontal[0, ksize // 2 + 1:] = 0
        tensor_mask_horizontal = tf.constant(mask_horizontal, dtype=tf.float32)
        self.W_horizontal = W_horizontal * tensor_mask_horizontal
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_horizontal), wd))

        self.W_v2h = tf.get_variable(name + '_v2h', shape=[1, 1, ch * 2, ch * 2],
                                          dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_v2h), wd))

        self.W_h2h = tf.get_variable(name + '_h2h', shape=[1, 1, ch, ch],
                                     dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_h2h), wd))

    def gated_conv(self, x_vertical, x_horizontal, isTrain,
             conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=False):
        # x_vertical   : [None, 29, 28, 64]
        # x_horizontal : [None, 28, 28, 64]
        self.L1_vertical = conv(x_vertical, self.W_vertical, isTrain,
         conv_stride=conv_stride, conv_padding=conv_padding, conv_dilation=conv_dilation, batch_norm=batch_norm,
                                activation=None)
        # L1_vertical   : [None, 29, 28, 128]

        self.L1_vertical_f, self.L1_vertical_g = tf.split(self.L1_vertical, 2)
        self.L1_vertical_f = tf.tanh(self.L1_vertical_f)
        self.L1_vertical_g = tf.sigmoid(self.L1_vertical_g)
        self.L2_vertical = self.L1_vertical_f * self.L1_vertical_g
        # L2_vertical   : [None, 29, 28, 64]

        self.L1_horizontal =  conv(x_horizontal, self.W_horizontal, isTrain,
         conv_stride=conv_stride, conv_padding=conv_padding, conv_dilation=conv_dilation, batch_norm=batch_norm,
                                   activation=None)
        # L1_horizontal : [None, 28, 28, 128]

        # crop bottom of the vertical stack(padded on top) by 1
        L1_vertical_croped = self.L1_vertical[:, :28, :, :]
        # L1_vertical_croped : [None, 28, 28, 128]

        self.L1_v2h = conv(L1_vertical_croped, self.W_v2h, isTrain,
         conv_stride=conv_stride, conv_padding=conv_padding, conv_dilation=conv_dilation, batch_norm=batch_norm,
                           activation=None)
        # L1_v2h : [None, 28, 28, 128]

        self.L2_horizontal = self.L1_horizontal + self.L1_v2h
        # L2_horizontal : [None, 28, 28, 128]

        self.L2_horizontal_f, self.L2_horizontal_g = tf.split(self.L2_horizontal, 2)
        self.L2_horizontal_f = tf.tanh(self.L2_horizontal_f)
        self.L2_horizontal_g = tf.sigmoid(self.L2_horizontal_g)
        self.L3_horizontal = self.L2_horizontal_f * self.L2_horizontal_g
        # L3_horizontal   : [None, 28, 28, 64]

        self.L4_horizontal = conv(self.L3_horizontal, self.W_h2h, isTrain,
         conv_stride=conv_stride, conv_padding=conv_padding, conv_dilation=conv_dilation, batch_norm=batch_norm,
                           activation=None)

        self.L5_horizontal = self.L4_horizontal + x_horizontal

        return self.L2_vertical, self.L5_horizontal


class resnet_block:
    def __init__(self, name, ch_input, ch_mid, ch_output, wd):
        self.ch_input = ch_input
        self.ch_mid = ch_mid
        self.ch_output = ch_output

        self.W1 = tf.get_variable(name + '_W1', shape=[1, 1, self.ch_input, self.ch_mid],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(0, tf.sqrt(2 / self.ch_input)))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W1), wd))

        self.W2 = tf.get_variable(name + '_W2', shape=[3, 3, self.ch_mid, self.ch_mid],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(0, tf.sqrt(2 / self.ch_mid)))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2), wd))

        self.W3 = tf.get_variable(name + '_W3', shape=[1, 1, self.ch_mid, self.ch_output],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(0, tf.sqrt(2 / self.ch_mid)))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3), wd))

        self.W_projection = tf.get_variable(name + '_W_projection', shape=[1, 1, self.ch_input, self.ch_output],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(0, tf.sqrt(2 / self.ch_input)))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_projection), wd))

    def make_residual_bottleneck_block(self, input, isTrain, downsampling=False):
        self.L1 = conv(input, self.W1, isTrain,
         conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=True, activation=tf.nn.relu)

        self.L2 = conv(self.L1, self.W2, isTrain,
                       conv_stride=(1, 2, 2, 1) if downsampling else (1, 1, 1, 1), conv_padding='SAME', conv_dilation=1,
                       batch_norm=True,
                       activation=tf.nn.relu)

        self.L3 = conv(self.L2, self.W3, isTrain,
         conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=True, activation=tf.nn.relu)

        self.L_projection = conv(input, self.W_projection, isTrain,
                                 conv_stride=(1, 2, 2, 1) if downsampling else (1, 1, 1, 1), conv_padding='SAME',
                                 conv_dilation=1, batch_norm=True,
                                 activation=tf.nn.relu)

        self.L4 = self.L3 + self.L_projection

        return self.L4