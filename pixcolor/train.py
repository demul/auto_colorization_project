import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

import model
import data_loader
import util


class ColorizationNet:
    def __init__(self, input_size, lr=0.000003, decaying_factor=0.0005,
                 adam_beta1=0.9, adam_beta2=0.99, result_dir='./results'):
        self.input_size = input_size
        self.lr = lr
        self.decaying_factor = decaying_factor
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.loss_sampling_step = None

        ### For plotting
        self.metric_list = dict()
        self.metric_list['losses'] = []

        ### For training and inference
        self.graph = tf.Graph()
        self.model = model.PixelCNN(input_size, decaying_factor=self.decaying_factor)
        self.data_loader = data_loader.DataLoader(validation_len=1000)
        self.decoder = util.Decoder(self.data_loader.table_gamut, temperature=0.38, sampling='probabilistic')


    def make_graph(self, luminance_, chrominance_, label_, isTrain, learning_rate_):
        logit = self.model.colorizer(luminance_, chrominance_, isTrain=isTrain)
        prob = tf.nn.softmax(logit)
        CEE = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=label_), axis=(1, 2)))
        tf.add_to_collection('losses', CEE)

        ##### must minimize total loss
        total_loss = tf.add_n(tf.get_collection('losses'))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_,
                                              beta1=self.adam_beta1, beta2=self.adam_beta2).minimize(total_loss)
        ##### but we need to watch Cross Entropy Error
        ##### to watch how well our model does converge.
        return train_op, prob, CEE


    def run(self, max_epoch, loss_sampling_step):
        self.loss_sampling_step = loss_sampling_step

        with self.graph.as_default():
            # declare placeholder
            L = tf.placeholder(shape=[None, 224, 224, 1], dtype=tf.float32)
            lb_ab = tf.placeholder(shape=[None, 28, 28, 2], dtype=tf.float32)
            lb_class = tf.placeholder(shape=[None, 28, 28, 262], dtype=tf.float32)
            isTrain = tf.placeholder(dtype=tf.bool)
            learning_rate = tf.placeholder(dtype=tf.float32)

            # make graph
            train_op, prob_, loss_ = self.make_graph(L, lb_ab, lb_class, isTrain, learning_rate)

            # define session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            # define saver
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state('./ckpt')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            # variable for calculating ptime
            start_time = time.time()
            for epoch in range(max_epoch):
                train_loss = 0
                val_loss = 0

                for itr in range(len(self.data_loader.idx_train)//self.input_size):
                    # load training data
                    input_batch, label_class_batch, label_ab_batch = self.data_loader.next_train(self.input_size)

                    # run session
                    _, loss = sess.run([train_op, loss_],
                                               feed_dict={L: input_batch, lb_class: label_class_batch,
                                                          lb_ab: label_ab_batch, isTrain: True, learning_rate: self.lr})
                    train_loss += loss / (len(self.data_loader.idx_train)//self.input_size)

                    # sample training loss
                    if itr % loss_sampling_step == 0:
                        progress_view = 'progress : ' \
                                        + '%7.6f' % (itr / (len(self.data_loader.idx_train)//self.input_size) * 100)\
                                        + '%  loss :' + '%7.6f' % loss
                        print(progress_view)
                        self.metric_list['losses'].append(loss)

                # save validation image
                input_batch, label_class_batch, label_ab_batch, GT_batch = self.data_loader.next_val(self.input_size)
                # put validation cursor back to 0
                self.data_loader.cursor_val = 0

                # get output ab
                output_batch_ab = self.recursive_image_generation(sess, L, lb_ab, isTrain, prob_, input_batch, label_ab_batch)
                # concat ab with L
                output_batch = np.concatenate([input_batch, output_batch_ab], axis=3)
                images_result_path = os.path.join(self.result_dir, 'epoch%04d.png' % (epoch + 1))
                self.show_result(output_batch[:8], GT_batch[:8], epoch + 1, save=True, path=images_result_path)

                # validate
                for itr in range(len(self.data_loader.idx_val)//self.input_size):
                    # load validation data
                    input_batch, label_class_batch, label_ab_batch, _ = self.data_loader.next_val(self.input_size)

                    # run session
                    loss = sess.run(loss_, feed_dict={L: input_batch, lb_class: label_class_batch,
                                                      lb_ab: label_ab_batch, isTrain: False})
                    val_loss += loss / (len(self.data_loader.idx_val)//self.input_size)

                with open('loss.txt', 'a') as wf:
                    epoch_time = time.time() - start_time
                    loss_info = '\nepoch: ' + '%7d' % (epoch + 1) + '  batch loss: %7.6f' % train_loss \
                                + '  val loss: %7.6f' % val_loss \
                                + '  time elapsed: ' + '%7.6f' % epoch_time
                    wf.write(loss_info)

                # save model by 10 epoch in different directory
                if epoch % 10 == 0 and epoch != 0:
                    model_dir = './ckpt' + '_epoch' + str(
                        epoch + 1) + '/model.ckpt'
                    saver.save(sess, model_dir)

                # save model
                model_dir = './ckpt' + '/model.ckpt'
                saver.save(sess, model_dir)

            # close session
            sess.close()
            # plot loss graph
            self.save_loss_plot()


    def recursive_image_generation(self, sess, L, lb_ab, isTrain, prob_, input_batch, label_ab_batch):
        # recursive multimodal sampling
        result_img = np.zeros([self.input_size, 28, 28, 2], dtype=tf.uint8)

        ## 'only a first pixel' is picked from label_ab_batch!
        ## and rest pixels are sampled recursively
        result_img[:, 0, 0, :] = label_ab_batch[:, 0, 0, :] # first (0, 0) index of result
        prob = sess.run(prob_, feed_dict={L: input_batch, lb_ab: label_ab_batch, isTrain: False})
        result_img[:, 0, 1, :] = self.decoder.encoding2ab(prob[:, 0, 1, :]) # second (0, 1) index of result

        ## make index list [(0, 2), (0, 3).....(27, 26), (27, 27)]
        list_idx_order = []
        for i in range(28):
            for j in range(28):
                if j == 0 or j == 1:
                    continue
                list_idx_order.append([i, j])

        for idx_pair in list_idx_order:
            prob = sess.run(prob_, feed_dict={L: input_batch, lb_ab: result_img, isTrain: False})
            result_img[:, idx_pair[0], idx_pair[1], :] = self.decoder.encoding2ab(prob[:, idx_pair[0], idx_pair[1], :])
        return result_img


    def show_result(self, img_gen, img_GT, num_epoch, show=False, save=False, path='result.png'):
        # cvt BGR to RGB
        img_gen = self.bgr2rgb(img_gen)
        img_GT = self.bgr2rgb(img_GT)

        size_figure_grid = 4
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
        for i in range(size_figure_grid) :
            for j in range (size_figure_grid) :
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

        for k in range(0, size_figure_grid * size_figure_grid, 2):
            i = k // size_figure_grid
            j = k % size_figure_grid
            ax[i, j].cla()
            ax[i, j].imshow(img_gen[k//2].astype(np.uint8))
            ax[i, j+1].cla()
            ax[i, j+1].imshow(img_GT[k//2].astype(np.uint8))

        label = 'Epoch %4d' % (num_epoch)
        fig.text(0.5, 0.04, label, ha='center')

        if save:
            plt.savefig(path)
        if show:
            plt.show()
        else:
            plt.close()


    def bgr2rgb(self, batch_img):
        b = batch_img[:, :, :, 0]
        g = batch_img[:, :, :, 1]
        r = batch_img[:, :, :, 2]
        return np.stack([r, g, b], axis=3)


    def save_loss_plot(self):
        x = range(1, self.loss_sampling_step * len(self.metric_list['losses']) + 1, self.loss_sampling_step)

        y1 = self.metric_list['losses']

        plt.plot(x, y1, label='loss')

        plt.xlabel('Iter')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        file_name = 'loss' + '.png'
        plt.savefig(file_name)
        plt.close()