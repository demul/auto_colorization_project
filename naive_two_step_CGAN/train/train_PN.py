import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time, os, pickle
import models.model_PN
import data_loader
import cv2

class CGAN_PN:
    def __init__(self, input_size, lr = 0.0002, smoothing_factor=0.2, input_dir = "yumi_cell", result_dir = 'PN_results'):
        self.lr = lr
        self.input_size = input_size
        self.smoothing_factor = smoothing_factor

        # model
        model = models.model_PN.model(input_size)
        self.generator = model.generator
        self.discriminator = model.discriminator

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
            ax[i, j].imshow(cv2.cvtColor(img_gen[k]/255, cv2.COLOR_RGB2BGR))
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
        x = range(len(hist['D_losses']))

        y1 = hist['D_losses']
        y2 = hist['G_losses']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')

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
        G_loss_list = []
        D_loss_list = []
        L1_loss_list = []

        iter_ = -1
        for d in range(gpu_num):
            with tf.device('/gpu:' + str(d)):
                for i in range(split_num):
                    iter_ += 1
                    with tf.variable_scope(tf.get_variable_scope()):
                        #####Generator
                        Generated_img = self.generator(LRC_list[iter_], edge_list[iter_], isTrain=is_training, reuse=iter_> 0, keep_prob=keep_prob)

                        #####Discriminator
                        D_real_logits = self.discriminator(GT_list[iter_], edge_list[iter_], isTrain=is_training, reuse=iter_ > 0)
                        D_fake_logits = self.discriminator(Generated_img, edge_list[iter_], isTrain=is_training, reuse=True)

                        #####Loss Functions
                        ## Add Label Smoothing (Discriminator를 혼동시키는 역할)
                        ## 여기선 0으로 세팅
                        loss_d_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits,
                                                                    labels=tf.ones_like(D_real_logits) * (
                                                                            1 - self.smoothing_factor) + self.smoothing_factor / self.input_size)
                        loss_d_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits,
                                                                    labels=tf.zeros_like(
                                                                        D_fake_logits) + self.smoothing_factor / self.input_size)
                        loss_d = loss_d_real + loss_d_fake

                        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

                        scale_factor_for_L1 = 1000
                        L1_loss= tf.reduce_mean(tf.abs(Generated_img - GT_list[iter_])) * scale_factor_for_L1

                        Generated_list.append(Generated_img)
                        G_loss_list.append(loss_g)
                        D_loss_list.append(loss_d)
                        L1_loss_list.append(L1_loss)

        total_Generated_img = tf.concat(Generated_list, axis=0)
        total_D_loss = tf.reduce_mean(D_loss_list, axis=0)
        total_G_loss = tf.reduce_mean(G_loss_list, axis=0)
        total_L1_loss = tf.reduce_mean(L1_loss_list, axis=0)

        target_G_loss = total_G_loss + total_L1_loss

        ##### Train Operation에 포함할 Weight들을 불러옴.
        vars_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='p_discriminator')
        vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='p_generator')

        ##### Optimizer and Train operation
        ### Operation을 아래와 같이 복잡하게 구성하는 이유
        ## 1. Update할 variable list를 분리해서 train_D엔 Discriminator 관련 Weight들만,
        ##    train_G엔 Generator 관련 Weight들만 Update되도록 해야함.
        ## 2. Batch Normalization parameter들은 tf.GraphKeys.TRAINABLE_VARIABLES에 저장되지 않음.
        ##    따라서 Batch Norm을 쓸려면 아래와 같이 tf.GraphKeys.UPDATE_OPS에 저장되어 있는 moving average를 불러와야 함.
        ## 3. colocate_gradients_with_ops=True 옵션을 주어 Forward가 일어났던 GPU에서 Gradient 계산도 일어나도록 처리.
        ##    http://openresearch.ai/t/tensorpack-multigpu/45
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op_d = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(tf.reduce_mean(total_D_loss), var_list=vars_d, colocate_gradients_with_ops=True)
            train_op_g = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(tf.reduce_mean(target_G_loss), var_list=vars_g, colocate_gradients_with_ops=True)

        return train_op_d, train_op_g, total_D_loss, total_G_loss, total_Generated_img

    def train(self, train_epoch, model_save_freq, gpu_num, split_num, train_size = 7000, test_size=300):
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
        train_op_d, train_op_g, loss_d, loss_g, Generated_img = self.make_train_graph(EDGE, GT, LRC,
                                                                                      is_training=is_training, keep_prob = keep_prob,
                                                                                      gpu_num=gpu_num, split_num=split_num)

        ##### Run Session
        sess = tf.Session()

        ##### Check the Checkpoint
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(os.path.join('./ckpt', 'ckpt_PN'))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())


        ##### Histogram for Visualizing
        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['times_per_epoch'] = []

        ##### Training Loop
        print("Trainig start!")
        for epoch in range(train_epoch):
            img_loader.shuffle()
            G_losses = []
            D_losses = []

            epoch_start_time = time.time()
            for iter in range((train_size + test_size)//self.input_size) :
                ######################################################################
                ######################################################################
                ####Get inputs
                ### Use LRC_batch as LRGT in training time !!!###
                LRC_batch = img_loader.next_LRC(self.input_size)
                # LRGT_batch = img_loader.next_LRGT(self.input_size)
                edge_batch, GT_batch = img_loader.next(self.input_size)
                ######################################################################
                ######################################################################

                ####Update D
                _, loss_D= sess.run ([train_op_d, loss_d],
                                    ######################################################################
                                    ############### Use LRC_batch as GT in training time !!!##############
                                    feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRC_batch, keep_prob: 0.3, is_training: True})
                                    # feed_dict = {EDGE: edge_batch, GT: GT_batch, LRC: LRGT_batch, keep_prob: 0.3, is_training: True})
                                    ######################################################################
                D_losses.append(loss_D)
                ####Updata G
                for i in range(3):
                    _, loss_G = sess.run([train_op_g, loss_g],
                                        ######################################################################
                                        ############### Use LRC_batch as GT in training time !!!##############
                                        feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRC_batch, keep_prob: 0.3, is_training: True})
                                        # feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRGT_batch, keep_prob: 0.3, is_training: True})
                                        ######################################################################
                G_losses.append(loss_G)


            ###Print Process
            epoch_end_time = time.time()
            time_per_epoch = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f'
                  % ((epoch + 1), train_epoch, time_per_epoch, np.mean(D_losses), np.mean(G_losses)))

            ######################################################################
            ######################################################################
            ############### Use LRC_batch as LRC in testing time !!!##############
            img_loader_test.cursor = 0
            LRC_batch = img_loader_test.next_LRC(self.input_size)
            edge_batch, GT_batch = img_loader_test.next(self.input_size)
            ######################################################################
            ######################################################################

            ##### 생성된 샘플이미지를 저장
            images_generated = sess.run(Generated_img, feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRC_batch,
                                                                  keep_prob: 1.0, is_training: False})

            images_result_path = os.path.join(self.result_dir, 'Fixed_noise_results', 'epoch%4d.png' % (epoch + 1))
            self.show_result(images_generated, GT_batch, (epoch + 1), save=True, path=images_result_path)

            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))
            train_hist['times_per_epoch'].append(time_per_epoch)

            if (epoch + 1) % model_save_freq == 0:
                model_dir = os.path.join('ckpt', 'ckpt_PN', '%4d_model.ckpt'%(epoch + 1))
                saver.save(sess, model_dir)
            else :
                model_dir = os.path.join('ckpt', 'ckpt_PN', 'model.ckpt')
                saver.save(sess, model_dir)


        print('Avg time per epoch: %.2f, total %d epochs ptime: %.2f'
              % (np.mean(train_hist['times_per_epoch']), train_epoch, np.sum(train_hist['times_per_epoch'])))
        print("Training finish!... save training results")

        with open(os.path.join(self.result_dir, 'train_hist.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)

        self.show_train_hist(train_hist, save=True, path= os.path.join(self.result_dir, 'train_hist.png'))

        sess.close()
        print('Finished!')


    def inference(self, gpu_num, split_num):
        # load image loader
        img_loader = data_loader.Loader(is_test=True, dir = self.input_dir)

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
        train_op_d, train_op_g, loss_d, loss_g, Generated_img = self.make_train_graph(EDGE, GT, LRC,
                                                                                      is_training=is_training,
                                                                                      keep_prob=keep_prob,
                                                                                      gpu_num=gpu_num,
                                                                                      split_num=split_num)
        ##### Run Session
        sess = tf.Session()

        ##### Check the Checkpoint
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(os.path.join('./ckpt', 'ckpt_PN'))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        for iter in range(300 // self.input_size):
            LRC_batch = img_loader.next_LRC(self.input_size)
            edge_batch, GT_batch = img_loader.next(self.input_size)
            ##### 생성된 샘플이미지를 저장
            images_generated = sess.run(Generated_img,
                                        feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRC_batch, keep_prob: 1.0, is_training: False})

            images_result_path = os.path.join(self.root, 'Fixed_noise_results', 'inference%4d.png' % (iter + 1))
            self.show_result(images_generated, GT_batch, (iter + 1), save=True, path=images_result_path)

        print("Inference finish!... save inference results")

        sess.close()
        print('Finished!')

    def save_final_result(self, gpu_num, split_num, start=0, end=7380):  # BD를 학습시키려면 LRC의 결과물이 필요하다.
        # load image loader
        img_loader = data_loader.Loader(dir=self.input_dir)

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
        train_op_d, train_op_g, loss_d, loss_g, Generated_img = self.make_train_graph(EDGE, GT, LRC,
                                                                                      is_training=is_training,
                                                                                      keep_prob=keep_prob,
                                                                                      gpu_num=gpu_num,
                                                                                      split_num=split_num)
        ##### Run Session
        sess = tf.Session()

        ##### Check the Checkpoint
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(os.path.join('./ckpt', 'ckpt_PN'))
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        images_result_dir = os.path.join(self.input_dir, 'PN')
        if not os.path.isdir(images_result_dir):
            os.mkdir(images_result_dir)

        img_loader.idx = range(start, end)  # 순서대로 불러오도록
        for iter in range((end - start) // self.input_size):
            LRC_batch = img_loader.next_LRC(self.input_size)
            edge_batch, GT_batch = img_loader.next(self.input_size)

            ##### 생성된 샘플이미지를 "모두" resize해서 저장
            images_generated = sess.run(Generated_img,
                                        feed_dict={EDGE: edge_batch, GT: GT_batch, LRC: LRC_batch, keep_prob: 1.0, is_training: False})
            for i in range(self.input_size):
                images_result_path = os.path.join(images_result_dir, '%4d.png' % (self.input_size * iter + i))
                cv2.imwrite(images_result_path, images_generated[i])

        print("High Resolution Colorizing finish!... save PN results")

        sess.close()
        print('Finished!')