import tensorflow as tf

class CIC_ConditioningNet:
    def __init__(self, input_size, decaying_factor=0.0005):
        self.input_size = input_size
        self.decaying_factor = decaying_factor

    def color_classifier(self, img, isTrain):
        # [None, 224, 224, 1]

        ###### conv1_1
        self.W1_1 = tf.get_variable('conv1_1', shape=[3, 3, 1, 64],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L1_1 = self.conv(img, self.W1_1, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1, batch_norm=False)
        # [None, 224, 224, 64]

        ###### conv1_2
        self.W1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L1_2 = self.conv(self.L1_1, self.W1_2, isTrain, conv_stride=(1, 2, 2, 1), conv_dilation=1, batch_norm=True)
        # [None, 112, 112, 64]

        ###### conv2_1
        self.W2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L2_1 = self.conv(self.L1_2, self.W2_1, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1, batch_norm=False)
        # [None, 112, 112, 128]

        ###### conv2_2
        self.W2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L2_2 = self.conv(self.L2_1, self.W2_2, isTrain, conv_stride=(1, 2, 2, 1), conv_dilation=1, batch_norm=True)
        # [None, 56, 56, 128]

        ###### conv3_1
        self.W3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L3_1 = self.conv(self.L2_2, self.W3_1, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1, batch_norm=False)
        # [None, 56, 56, 256]

        ###### conv3_2
        self.W3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L3_2 = self.conv(self.L3_1, self.W3_2, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1, batch_norm=False)
        # [None, 56, 56, 256]

        ###### conv3_3
        self.W3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L3_3 = self.conv(self.L3_2, self.W3_3, isTrain, conv_stride=(1, 2, 2, 1), conv_dilation=1, batch_norm=True)
        # [None, 28, 28, 256]

        ###### conv4_1
        self.W4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L4_1 = self.conv(self.L3_3, self.W4_1, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1,
                              batch_norm=False)
        # [None, 28, 28, 512]

        ###### conv4_2
        self.W4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L4_2 = self.conv(self.L4_1, self.W4_2, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1,
                              batch_norm=False)
        # [None, 28, 28, 512]

        ###### conv4_3
        self.W4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L4_3 = self.conv(self.L4_2, self.W4_3, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1, batch_norm=True)
        # [None, 28, 28, 512]

        ###### conv5_1
        self.W5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L5_1 = self.conv(self.L4_3, self.W5_1, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=2,
                              batch_norm=False)
        # [None, 28, 28, 512]

        ###### conv5_2
        self.W5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L5_2 = self.conv(self.L5_1, self.W5_2, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=2,
                              batch_norm=False)
        # [None, 28, 28, 512]

        ###### conv5_3
        self.W5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L5_3 = self.conv(self.L5_2, self.W5_3, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=2, batch_norm=True)
        # [None, 28, 28, 512]

        ###### conv6_1
        self.W6_1 = tf.get_variable('conv6_1', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L6_1 = self.conv(self.L5_3, self.W6_1, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=2,
                              batch_norm=False)
        # [None, 28, 28, 512]

        ###### conv6_2
        self.W6_2 = tf.get_variable('conv6_2', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L6_2 = self.conv(self.L6_1, self.W6_2, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=2,
                              batch_norm=False)
        # [None, 28, 28, 512]

        ###### conv6_3
        self.W6_3 = tf.get_variable('conv6_3', shape=[3, 3, 512, 512],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L6_3 = self.conv(self.L6_2, self.W6_3, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=2, batch_norm=True)
        # [None, 28, 28, 512]

        ###### conv7_1
        self.W7_1 = tf.get_variable('conv7_1', shape=[3, 3, 512, 256],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L7_1 = self.conv(self.L6_3, self.W7_1, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1,
                              batch_norm=False)
        # [None, 28, 28, 256]

        ###### conv7_2
        self.W7_2 = tf.get_variable('conv7_2', shape=[3, 3, 256, 256],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L7_2 = self.conv(self.L7_1, self.W7_2, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1,
                              batch_norm=False)
        # [None, 28, 28, 256]

        ###### conv7_3
        self.W7_3 = tf.get_variable('conv7_3', shape=[3, 3, 256, 256],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L7_3 = self.conv(self.L7_2, self.W7_3, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1, batch_norm=True)
        # [None, 28, 28, 256]

        ###### conv8_1
        ###### [[[[upsample]]]
        self.W8_1 = tf.get_variable('conv8_1', shape=[3, 3, 128, 256],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L8_1 = tf.nn.conv2d_transpose(self.L7_3, self.W8_1, output_shape=[self.input_size, 56, 56, 128],
                                          strides=[1, 2, 2, 1], padding='SAME')
        self.L8_1 = tf.nn.relu(self.L8_1)
        # [None, 56, 56, 128]

        ###### conv8_2
        self.W8_2 = tf.get_variable('conv8_2', shape=[3, 3, 128, 128],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L8_2 = self.conv(self.L8_1, self.W8_2, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1,
                              batch_norm=False)
        # [None, 56, 56, 128]

        ###### conv8_3
        self.W8_3 = tf.get_variable('conv8_3', shape=[3, 3, 128, 128],
                                    dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.L8_3 = self.conv(self.L8_2, self.W8_3, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1,
                              batch_norm=False)
        # [None, 56, 56, 128]

        ###### conv9 (1x1 conv)
        self.W9 = tf.get_variable('conv9', shape=[1, 1, 128, 262],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.01))
        self.logit = self.conv(self.L8_3, self.W9, isTrain, conv_stride=(1, 1, 1, 1), conv_dilation=1, batch_norm=False,
                               activation=None)
        # [None, 56, 56, 262]

        ##### collect weights for weight decay
        ##### weight decay is the methods for giving panalty to too-big weights which is likely to induce co-adaptations.
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W1_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W1_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3_3), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W4_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W4_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W4_3), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W5_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W5_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W5_3), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W6_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W6_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W6_3), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W7_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W7_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W7_3), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W8_1), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W8_2), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W8_3), self.decaying_factor))
        tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W9), self.decaying_factor))

        return self.logit, self.L8_2


    def conv(self, x, w, isTrain,
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
