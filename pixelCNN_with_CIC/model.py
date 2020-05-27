import tensorflow as tf
import ops
from model_CIC import CIC_ConditioningNet


class PixelCNN:
    def __init__(self, input_size, decaying_factor=0.0005):
        self.input_size = input_size
        self.decaying_factor = decaying_factor
        self.conditioning_net = CIC_ConditioningNet(self.input_size, decaying_factor=0)

    def colorizer(self, luminance, chrominance, isTrain):
        # luminance : [None, 224, 224, 1]
        # luminance : [None, 56, 56, 2]

        logit_class, condition = self.conditioning_net.color_classifier(luminance, isTrain)
        # logit_class : [None, 56, 56, 262]
        # condition : [None, 56, 56, 128]

        luminance_resized = tf.image.resize(luminance, [56, 56])
        # luminance_resized : [None, 56, 56, 1]


        # concat condition with luminance
        condition = tf.concat([condition, luminance_resized], axis=3)
        # input_adaptation : [None, 56, 56, 129]


        with tf.variable_scope('ColorizationNet'):
            ###### conv1
            self.W1 = tf.get_variable('conv1', shape=[7, 7, 2, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            # weight decay is done in this function
            self.L1 = ops.masked_conv_typeA(chrominance, self.W1, isTrain, self.decaying_factor,
                                  conv_stride=(1, 1, 1, 1), conv_padding='SAME',
                                  conv_dilation=1, batch_norm=False, activation=tf.nn.relu)
            # [None, 56, 56, 64]

            ####################################################################################
            # concat ab-dependent, A-masked-conved feature map and global conditioning feature map
            self.L_conditioned1 = tf.concat([self.L1, condition], axis=3)
            # [None, 56, 56, 129]
            self.W_fuse_condition1 = tf.get_variable('fuse_condition1', shape=[1, 1, 193, 256],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_fuse_condition1), self.decaying_factor))
            self.L_conditioned2 = ops.conv(self.L_conditioned1, self.W_fuse_condition1, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=True, activation=tf.nn.relu)
            # [None, 56, 56, 256]

            self.W_fuse_condition2 = tf.get_variable('fuse_condition2', shape=[1, 1, 256, 1024],
                                                     dtype=tf.float32,
                                                     initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_fuse_condition2), self.decaying_factor))
            self.L_conditioned3 = ops.conv(self.L_conditioned2, self.W_fuse_condition2, isTrain,
                                           conv_stride=(1, 1, 1, 1),
                                           conv_dilation=1, batch_norm=True, activation=tf.nn.relu)
            # [None, 56, 56, 1024]

            self.W_fuse_condition3 = tf.get_variable('fuse_condition3', shape=[1, 1, 1024, 64],
                                                     dtype=tf.float32,
                                                     initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W_fuse_condition3), self.decaying_factor))
            self.L_conditioned4 = ops.conv(self.L_conditioned3, self.W_fuse_condition3, isTrain,
                                           conv_stride=(1, 1, 1, 1),
                                           conv_dilation=1, batch_norm=True, activation=tf.nn.relu)
            # [None, 56, 56, 256]
            ####################################################################################

            L1_vertical, L1_horizontal= ops.split_into_stacks(self.L_conditioned4)
            # L1_vertical [None, 57, 56, 64], L1_horizontal [None, 56, 56, 64]

            ###### gated conv unit x 10
            for i in range(10):
                gated_conv_block = ops.gated_conv_unit('gated_conv%d' % i, self.decaying_factor, ch=64, ksize=5)
                L1_vertical, L1_horizontal= gated_conv_block.gated_conv(L1_vertical, L1_horizontal, isTrain,
                                 conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=False)
            # [None, 56, 56, 64]

            ###### conv2
            self.W2 = tf.get_variable('conv2', shape=[1, 1, 64, 1024],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2), self.decaying_factor))
            self.L2 = ops.conv(L1_horizontal, self.W2, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=True, activation=tf.nn.relu)
            # [None, 56, 56, 1024]

            ###### conv3
            self.W3 = tf.get_variable('conv3', shape=[1, 1, 1024, 262],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3), self.decaying_factor))
            self.L3 = ops.conv(self.L2, self.W3, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=True, activation=None)
            # [None, 56, 56, 262]

            self.logit = self.L3 + logit_class

        return self.logit, logit_class
