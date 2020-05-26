import tensorflow as tf
import ops

class ConditioningNet:
    def __init__(self, input_size, decaying_factor=0.0005):
        self.input_size = input_size
        self.decaying_factor = decaying_factor

    def conditioner(self, img, isTrain):
        # [None, 224, 224, 1]
        with tf.variable_scope('ConditioningNet'):
            ###### conv1
            self.W1 = tf.get_variable('conv1', shape=[7, 7, 1, 64],
                                        dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W1), self.decaying_factor))
            self.L1 = ops.conv(img, self.W1, isTrain, conv_stride=(1, 2, 2, 1), conv_dilation=1, batch_norm=True)
            # [None, 112, 112, 64]

            ###### resnet block 1
            for i in range(3):
                res_block = ops.resnet_block('block1_%d' % i, 64 if i == 0 else 256, 64, 256, self.decaying_factor)
                self.L1 = res_block.make_residual_bottleneck_block(self.L1, isTrain, downsampling=True if i == 0 else False)
            # [None, 56, 56, 256]

            ###### resnet block 2
            for i in range(4):
                res_block = ops.resnet_block('block2_%d' % i, 256 if i == 0 else 512, 128, 512, self.decaying_factor)
                self.L1 = res_block.make_residual_bottleneck_block(self.L1, isTrain, downsampling=True if i == 0 else False)
            # [None, 28, 28, 512]

            ###### resnet block 3
            for i in range(23):
                res_block = ops.resnet_block('block3_%d' % i, 512 if i == 0 else 1024, 256, 1024, self.decaying_factor)
                self.L1 = res_block.make_residual_bottleneck_block(self.L1, isTrain, downsampling=False)
            # [None, 28, 28, 1024]

            ###### conv2
            self.W2 = tf.get_variable('conv2', shape=[3, 3, 1024, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2), self.decaying_factor))
            self.L2 = ops.conv(self.L1, self.W2, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=True)
            # [None, 28, 28, 64]

            ###### conv3
            self.W3 = tf.get_variable('conv3', shape=[3, 3, 64, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3), self.decaying_factor))
            self.L3 = ops.conv(self.L2, self.W3, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=True)
            # [None, 28, 28, 64]

            ###### conv4
            self.W4 = tf.get_variable('conv4', shape=[3, 3, 64, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W4), self.decaying_factor))
            self.L4 = ops.conv(self.L3, self.W4, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=False, activation=None)
            # [None, 28, 28, 64]

        return self.L4

class AdaptationNet:
    def __init__(self, input_size, decaying_factor=0.0005):
        self.input_size = input_size
        self.decaying_factor = decaying_factor

    def adapter(self, condition, isTrain):
        # [None, 28, 28, 65]
        with tf.variable_scope('AdaptationNet'):
            ###### conv1
            self.W1 = tf.get_variable('conv1', shape=[3, 3, 65, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W1), self.decaying_factor))
            self.L1 = ops.conv(condition, self.W1, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=True)
            # [None, 28, 28, 64]

            ###### conv3
            self.W2 = tf.get_variable('conv2', shape=[3, 3, 64, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2), self.decaying_factor))
            self.L2 = ops.conv(self.L1, self.W2, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=True)
            # [None, 28, 28, 64]

            ###### conv3
            self.W3 = tf.get_variable('conv3', shape=[3, 3, 64, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3), self.decaying_factor))
            self.L3 = ops.conv(self.L2, self.W3, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=False, activation=None)
            # [None, 28, 28, 64]

            return self.L3


class PixelCNN:
    def __init__(self, input_size, decaying_factor=0.0005):
        self.input_size = input_size
        self.decaying_factor = decaying_factor

    def colorizer(self, luminance, chrominance, isTrain):
        # luminance : [None, 224, 224, 1]
        # luminance : [None, 28, 28, 2]

        conditioning_net = ConditioningNet(self.input_size, self.decaying_factor)
        condition = conditioning_net.conditioner(luminance, isTrain)
        # condition : [None, 28, 28, 64]

        luminance_resized = tf.image.resize(luminance, [28, 28])
        # luminance_resized : [None, 28, 28, 1]

        input_adaptation = tf.concat([condition, luminance_resized], axis=3)
        # input_adaptation : [None, 28, 28, 65]

        adaptation_net = AdaptationNet(self.input_size, self.decaying_factor)
        condition_adapted = adaptation_net.adapter(input_adaptation, isTrain)
        # condition_adapted : [None, 28, 28, 64]

        self.condition_with_chrominance = tf.concat([condition_adapted, chrominance], axis=3)
        # condition_adapted : [None, 28, 28, 66]

        with tf.variable_scope('ColorizationNet'):
            ###### conv1
            self.W1 = tf.get_variable('conv1', shape=[7, 7, 66, 64],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            # weight decay is done in this function
            self.L1 = ops.masked_conv_typeA(self.condition_with_chrominance, self.W1, isTrain, self.decaying_factor,
                                  conv_stride=(1, 1, 1, 1), conv_padding='SAME',
                                  conv_dilation=1, batch_norm=False, activation=tf.nn.relu)
            # [None, 28, 28, 64]

            L1_vertical, L1_horizontal= ops.split_into_stacks(self.L1)
            # L1_vertical [None, 29, 28, 64], L1_horizontal [None, 28, 28, 64]

            ###### gated conv unit x 10
            for i in range(10):
                gated_conv_block = ops.gated_conv_unit('gated_conv%d' % i, self.decaying_factor, ch=64, ksize=5)
                L1_vertical, L1_horizontal= gated_conv_block.gated_conv(L1_vertical, L1_horizontal, isTrain,
                                 conv_stride=(1, 1, 1, 1), conv_padding='SAME', conv_dilation=1, batch_norm=False)
            # [None, 28, 28, 64]

            ###### conv2
            self.W2 = tf.get_variable('conv2', shape=[1, 1, 64, 1024],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W2), self.decaying_factor))
            self.L2 = ops.conv(L1_horizontal, self.W2, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=False, activation=tf.nn.relu)
            # [None, 28, 28, 1024]

            ###### conv3
            self.W3 = tf.get_variable('conv3', shape=[1, 1, 1024, 262],
                                      dtype=tf.float32, initializer=tf.truncated_normal_initializer(0, 0.1))
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(self.W3), self.decaying_factor))
            self.logit = ops.conv(self.L2, self.W3, isTrain, conv_stride=(1, 1, 1, 1),
                               conv_dilation=1, batch_norm=False, activation=None)
            # [None, 28, 28, 262]

            return self.logit
