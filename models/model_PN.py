import numpy as np
import tensorflow as tf

class model:
    def __init__(self, input_size, total_split=4):
        self.input_size = input_size // total_split

    def generator(self, x, outline, isTrain=True, keep_prob=0.5, reuse=None):
        # x = Low-Resolution-Image [None, 256, 256, 3]
        # outline [None, 256, 256, 1]
        # drop-out은 regularizer로만 사용. decoding network에 적용


        x = tf.concat([x, outline], axis=3)  # [None, 256, 256, 5]

        with tf.variable_scope('p_generator', reuse=reuse):

            ###### 1st conv
            self.W1 = tf.get_variable("G_conv_w1", shape=[4, 4, 4, 64], dtype=np.float32,
                                        initializer=tf.random_normal_initializer(0, 0.01))
            self.L1 = tf.nn.conv2d(x, self.W1, strides=[1, 2, 2, 1], padding='SAME') # [None, 128, 128, 64]
            self.L1 = tf.nn.leaky_relu(self.L1)  # default alpha of leaky relu = 0.2
                                                 # the first layer doesn't have BN

            ###### 2st conv
            self.W2 = tf.get_variable("G_conv_w2", shape=[4, 4, 64, 128], dtype=np.float32,
                                      initializer=tf.random_normal_initializer(0, 0.01))
            self.L2 = tf.nn.conv2d(self.L1, self.W2, strides=[1, 2, 2, 1], padding='SAME') # [None, 64, 64, 128]
            self.L2 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.L2, training=isTrain))

            ###### 3st conv
            self.W3 = tf.get_variable("G_conv_w3", shape=[4, 4, 128, 256], dtype=np.float32,
                                      initializer=tf.random_normal_initializer(0, 0.01))
            self.L3 = tf.nn.conv2d(self.L2, self.W3, strides=[1, 2, 2, 1], padding='SAME') # [None, 32, 32, 256]
            self.L3 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.L3, training=isTrain))

            ###### 4st conv
            self.W4 = tf.get_variable("G_conv_w4", shape=[4, 4, 256, 512], dtype=np.float32,
                                      initializer=tf.random_normal_initializer(0, 0.01))
            self.L4 = tf.nn.conv2d(self.L3, self.W4, strides=[1, 2, 2, 1], padding='SAME') # [None, 16, 16, 512]
            self.L4 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.L4, training=isTrain))

            ###### 5st conv
            self.W5 = tf.get_variable("G_conv_w5", shape=[4, 4, 512, 512], dtype=np.float32,
                                      initializer=tf.random_normal_initializer(0, 0.01))
            self.L5 = tf.nn.conv2d(self.L4, self.W5, strides=[1, 2, 2, 1], padding='SAME')  # [None, 8, 8, 512]
            self.L5 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.L5, training=isTrain))

            ###### 6st conv
            self.W6 = tf.get_variable("G_conv_w6", shape=[4, 4, 512, 512], dtype=np.float32,
                                      initializer=tf.random_normal_initializer(0, 0.01))
            self.L6 = tf.nn.conv2d(self.L5, self.W6, strides=[1, 2, 2, 1], padding='SAME')  # [None, 4, 4, 512]
            self.L6 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.L6, training=isTrain))

            ###### 7st conv
            self.W7 = tf.get_variable("G_conv_w7", shape=[4, 4, 512, 512], dtype=np.float32,
                                      initializer=tf.random_normal_initializer(0, 0.01))
            self.L7 = tf.nn.conv2d(self.L6, self.W7, strides=[1, 2, 2, 1], padding='SAME')  # [None, 2, 2, 512]
            self.L7 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.L7, training=isTrain))

            ###### 8st conv
            self.W8 = tf.get_variable("G_conv_w8", shape=[4, 4, 512, 512], dtype=np.float32,
                                      initializer=tf.random_normal_initializer(0, 0.01))
            self.L8 = tf.nn.conv2d(self.L7, self.W8, strides=[1, 2, 2, 1], padding='SAME')  # [None, 1, 1, 512]
            self.L8 = tf.nn.relu(tf.layers.batch_normalization(self.L8, training=isTrain))

            ###### 1st upconv
            self.uW1 = tf.get_variable("G_uconv_w1", shape=[4, 4, 512, 512], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL1 = tf.nn.conv2d_transpose(self.L8, self.uW1, output_shape=[self.input_size, 2, 2, 512],
                                              strides=[1, 2, 2, 1], padding='SAME')
            self.uL1 = tf.nn.relu(tf.nn.dropout(tf.layers.batch_normalization(
                self.uL1, training=isTrain), keep_prob=keep_prob))
            # drop-out은 regularizer로서 학습에만 사용

            # skip connection
            self.uL1 = tf.concat([self.uL1, self.L7], axis=3)  # [None, 2, 2, 1024]

            ###### 2st upconv
            self.uW2 = tf.get_variable("G_uconv_w2", shape=[4, 4, 512, 1024], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL2 = tf.nn.conv2d_transpose(self.uL1, self.uW2, output_shape=[self.input_size, 4, 4, 512],
                                              strides=[1, 2, 2, 1], padding='SAME')
            self.uL2 = tf.nn.relu(tf.nn.dropout(tf.layers.batch_normalization(
                self.uL2, training=isTrain), keep_prob=keep_prob))
            # drop-out은 regularizer로서 학습에만 사용

            # skip connection
            self.uL2 = tf.concat([self.uL2, self.L6], axis=3)  # [None, 4, 4, 1024]

            ###### 3st upconv
            self.uW3 = tf.get_variable("G_uconv_w3", shape=[4, 4, 512, 1024], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL3 = tf.nn.conv2d_transpose(self.uL2, self.uW3, output_shape=[self.input_size, 8, 8, 512],
                                              strides=[1, 2, 2, 1], padding='SAME')
            self.uL3 = tf.nn.relu(tf.nn.dropout(tf.layers.batch_normalization(
                self.uL3, training=isTrain), keep_prob=keep_prob))
            # drop-out은 regularizer로서 학습에만 사용

            # skip connection
            self.uL3 = tf.concat([self.uL3, self.L5], axis=3)  # [None, 8, 8, 1024]

            ###### 4st upconv
            self.uW4 = tf.get_variable("G_uconv_w4", shape=[4, 4, 512, 1024], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL4 = tf.nn.conv2d_transpose(self.uL3, self.uW4, output_shape=[self.input_size, 16, 16, 512],
                                              strides=[1, 2, 2, 1], padding='SAME')
            self.uL4 = tf.nn.relu(tf.layers.batch_normalization(self.uL4, training=isTrain))
            # skip connection
            self.uL4 = tf.concat([self.uL4, self.L4], axis=3)  # [None, 16, 16, 1024]

            ###### 5st upconv
            self.uW5 = tf.get_variable("G_uconv_w5", shape=[4, 4, 256, 1024], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL5 = tf.nn.conv2d_transpose(self.uL4, self.uW5, output_shape=[self.input_size, 32, 32, 256],
                                              strides=[1, 2, 2, 1], padding='SAME')
            self.uL5 = tf.nn.relu(tf.layers.batch_normalization(self.uL5, training=isTrain))
            # skip connection
            self.uL5 = tf.concat([self.uL5, self.L3], axis=3)  # [None, 32, 32, 512]

            ###### 6st upconv
            self.uW6 = tf.get_variable("G_uconv_w6", shape=[4, 4, 128, 512], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL6 = tf.nn.conv2d_transpose(self.uL5, self.uW6, output_shape=[self.input_size, 64, 64, 128],
                                              strides=[1, 2, 2, 1], padding='SAME')
            self.uL6 = tf.nn.relu(tf.layers.batch_normalization(self.uL6, training=isTrain))
            # skip connection
            self.uL6 = tf.concat([self.uL6, self.L2], axis=3)  # [None, 64, 64, 256]

            ###### 7st upconv
            self.uW7 = tf.get_variable("G_uconv_w7", shape=[4, 4, 64, 256], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL7 = tf.nn.conv2d_transpose(self.uL6, self.uW7, output_shape=[self.input_size, 128, 128, 64],
                                              strides=[1, 2, 2, 1], padding='SAME')
            self.uL7 = tf.nn.relu(tf.layers.batch_normalization(self.uL7, training=isTrain))
            # skip connection
            self.uL7 = tf.concat([self.uL7, self.L1], axis=3)  # [None, 128, 128, 128]

            ###### 8st upconv
            self.uW8 = tf.get_variable("G_uconv_w8", shape=[4, 4, 3, 128], dtype=np.float32,
                                       initializer=tf.random_normal_initializer(0, 0.01))
            self.uL8 = tf.nn.conv2d_transpose(self.uL7, self.uW8, output_shape=[self.input_size, 256, 256, 3],
                                              strides=[1, 2, 2, 1], padding='SAME')

            return tf.nn.tanh(self.uL8) * 255

    def discriminator(self, x, unknown, isTrain=True, reuse=None):
        x = tf.concat([x, unknown], axis=3)
        # x = High-Resolution-Image + Outline [None, 256, 256, 3 + 1]

        with tf.variable_scope('p_discriminator', reuse=reuse):
            # 1st conv
            self.d_W1 =tf.get_variable("D_conv_w1", shape = [4, 4, 4, 64], dtype=np.float32,
                                       initializer = tf.random_normal_initializer(0, 0.01))
            self.d_L1 = tf.nn.conv2d(x, self.d_W1, strides=[1, 2, 2, 1], padding='SAME') # [None, 128, 128, 64]
            self.d_L1 = tf.nn.leaky_relu(self.d_L1) #default alpha of leaky relu = 0.2

            # 2st conv
            self.d_W2 = tf.get_variable("D_conv_w2", shape = [4, 4, 64, 128], dtype=np.float32,
                                        initializer=tf.random_normal_initializer(0, 0.01))
            self.d_L2 = tf.nn.conv2d(self.d_L1, self.d_W2, strides=[1, 2, 2, 1], padding='SAME') # [None, 64, 64, 128]
            self.d_L2 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.d_L2, training=isTrain))

            # 3st conv
            self.d_W3 = tf.get_variable("D_conv_w3", shape=[1, 1, 128, 256], dtype=np.float32,
                                        initializer=tf.random_normal_initializer(0, 0.01))
            self.d_L3 = tf.nn.conv2d(self.d_L2, self.d_W3, strides=[1, 2, 2, 1], padding='SAME')  # [None, 32, 32, 256]
            self.d_L3 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.d_L3, training=isTrain))

            # 4st conv
            self.d_W4 = tf.get_variable("D_conv_w4", shape=[4, 4, 256, 512], dtype=np.float32,
                                        initializer=tf.random_normal_initializer(0, 0.01))
            self.d_L4 = tf.nn.conv2d(self.d_L3, self.d_W4, strides=[1, 2, 2, 1], padding='SAME')  # [None, 16, 16, 512]
            self.d_L4 = tf.nn.leaky_relu(tf.layers.batch_normalization(self.d_L4, training=isTrain))

            # 5st conv
            self.d_W5 = tf.get_variable("D_11conv", shape=[1, 1, 512, 1], dtype=np.float32,
                                        initializer=tf.random_normal_initializer(0, 0.01))
            self.d_L5 = tf.nn.conv2d(self.d_L4, self.d_W5, strides=[1, 2, 2, 1], padding='SAME')  # [None, 8, 8, 1]

            return tf.squeeze(tf.reduce_mean(self.d_L5, axis = [1, 2]))