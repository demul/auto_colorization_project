import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time, os, pickle
import models.model_BD
import data_loader
import cv2

class BD:
    def __init__(self, input_size, lr = 0.0002, smoothing_factor=0.2, input_dir = "yumi_cell", result_dir = 'BD_results'):
        self.lr = lr
        self.input_size = input_size
        self.smoothing_factor = smoothing_factor #필요없음

        # model
        model = models.model_BD.model(input_size)
        self.generator = model.background_detector

        # results save folder
        self.input_dir = input_dir
        self.result_dir = result_dir
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        if not os.path.isdir(os.path.join(self.result_dir, 'Fixed_noise_results')):
            os.mkdir(os.path.join(self.result_dir, 'Fixed_noise_results'))


    def show_result(self, img_gen, img_GT, num_epoch, show=False, save=False, path='result.png'):
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
            ax[i, j].imshow(np.squeeze(img_gen[k]))
            ax[i, j+1].cla()
            ax[i, j+1].imshow(cv2.cvtColor(img_GT[k]/255, cv2.COLOR_RGB2BGR))

        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')

        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()

    def show_train_hist(self, hist, show=False, save=False, path='Train_hist.png'):
        x = range(len(hist['L1_losses']))

        y1 = hist['L1_losses']

        plt.plot(x, y1, label='L1_losses')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()

        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()

    def make_train_graph(self, edge_img, GT_img, LRC_img, is_training, keep_prob, gpu_num, split_num):
        edge_list = tf.split(edge_img, gpu_num * split_num, axis=0)
        GT_list = tf.split(GT_img, gpu_num * split_num, axis=0)
        LRC_list = tf.split(LRC_img, gpu_num * split_num, axis=0)

        Generated_list = []
        L1_loss_list = []
        iter_ = -1
        for d in range(gpu_num):
            with tf.device('/gpu:' + str(d)):
                for i in range(split_num):
                    iter_ += 1
                    with tf.variable_scope(tf.get_variable_scope()):
                        #####Generator
                        Generated_img, L1_loss = self.generator(edge_list[iter_], GT_list[iter_], LRC_list[iter_], isTrain=is_training, reuse=iter_ > 0, keep_prob=keep_prob)

                        Generated_list.append(Generated_img)
                        L1_loss_list.append(L1_loss)

        total_Generated_img = tf.concat(Generated_list, axis=0)
        total_L1_loss = tf.reduce_mean(L1_loss_list, axis=0)

        ##### Train Operation에 포함할 Weight들을 불러옴.
        vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='background_detector')

        ##### Optimizer and Train operation
        ### Operation을 아래와 같이 복잡하게 구성하는 이유
        ## 2. Batch Normalization parameter들은 tf.GraphKeys.TRAINABLE_VARIABLES에 저장되지 않음.
        ##    따라서 Batch Norm을 쓸려면 아래와 같이 tf.GraphKeys.UPDATE_OPS에 저장되어 있는 moving average를 불러와야 함.
        ## 3. colocate_gradients_with_ops=True 옵션을 주어 Forward가 일어났던 GPU에서 Gradient 계산도 일어나도록 처리.
        ##    http://openresearch.ai/t/tensorpack-multigpu/45
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(tf.reduce_mean(total_L1_loss), var_list=vars_g, colocate_gradients_with_ops=True)

        return train_op_g, total_L1_loss, total_Generated_img

    def train(self, train_epoch, model_save_freq, gpu_num, split_num, train_size=5800, test_size=100):
        #load image loader
        img_loader = data_loader.Loader(dir = self.input_dir, train_size=train_size, test_size=test_size)
        img_loader_test = data_loader.Loader(is_test=True, dir = self.input_dir, train_size=train_size, test_size=test_size)

        #####Make placeholder
        ### edge
        EDGE = tf.placeholder(tf.float32, [None, 256, 256, 1])
        ### GT
        GT = tf.placeholder(tf.float32, [None, 256, 256, 3])
        ### LRC
        LRC = tf.placeholder(tf.float32, [None, 256, 256, 3])

        ### Keep_prob
        keep_prob = tf.placeholder(tf.float32)
        ### Is training
        is_training = tf.placeholder(tf.bool)

        ##### Make Graph
        train_op_g, loss_L1, Generated_img = self.make_train_graph(EDGE, GT, LRC, is_training=is_training, keep_prob = keep_prob,
                                                                   gpu_num=gpu_num, split_num=split_num)

        ##### Run Session
        sess = tf.Session()

        ##### Check the Checkpoint
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(os.path.join('./ckpt', 'ckpt_BD'))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())


        ##### Histogram for Visualizing
        train_hist = {}
        train_hist['L1_losses'] = []
        train_hist['times_per_epoch'] = []

        ##### Training Loop
        for epoch in range(train_epoch):
            img_loader.shuffle()
            L1_losses = []

            epoch_start_time = time.time()
            for iter in range((train_size + test_size)//self.input_size) :
                ####get inputs
                LRC_batch = img_loader.next_LRC(self.input_size)
                edge_batch, GT_batch = img_loader.next(self.input_size)

                ####Update
                _, loss_L1_val = sess.run ([train_op_g, loss_L1],
                                      feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRC_batch, keep_prob: 0.5, is_training: True})
                L1_losses.append(loss_L1_val)


            ###Print Process
            epoch_end_time = time.time()
            time_per_epoch = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_L1: %.3f'
                  % ((epoch + 1), train_epoch, time_per_epoch, np.mean(L1_losses)))

            img_loader_test.cursor = 0
            LRC_batch = img_loader_test.next_LRC(self.input_size)
            edge_batch, GT_batch = img_loader_test.next(self.input_size)

            ##### 생성된 샘플이미지를 저장
            images_generated = sess.run(Generated_img,
                                        feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRC_batch, keep_prob: 1.0, is_training: False}) #여기선 drop out을 regularizer로 사용

            images_result_path = os.path.join(self.result_dir, 'Fixed_noise_results', 'epoch%4d.png' % (epoch + 1))
            self.show_result(images_generated, GT_batch, (epoch + 1), save=True, path=images_result_path)

            train_hist['L1_losses'].append(np.mean(L1_losses))
            train_hist['times_per_epoch'].append(time_per_epoch)

            if (epoch + 1) % model_save_freq == 0:
                model_dir = os.path.join('ckpt', 'ckpt_BD', '%4d_model.ckpt'%(epoch + 1))
                saver.save(sess, model_dir)
            else :
                model_dir = os.path.join('ckpt', 'ckpt_BD', 'model.ckpt')
                saver.save(sess, model_dir)


        print('Avg time per epoch: %.2f, total %d epochs ptime: %.2f'
              % (np.mean(train_hist['times_per_epoch']), train_epoch, np.sum(train_hist['times_per_epoch'])))
        print("Training finish!... save training results")

        with open(os.path.join(self.result_dir, 'train_hist.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)

        self.show_train_hist(train_hist, save=True, path= os.path.join(self.result_dir, 'train_hist.png'))

        sess.close()
        print('Finished!')