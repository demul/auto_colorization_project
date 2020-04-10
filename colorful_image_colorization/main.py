import data_preprocessor
import train
import os

###########################################################
# differences with reference paper(Colorful Image Colorization) :
# my dataset is 'MS-COCO'
# my initial learning rate(hyper parameter) is 1/10 of reference paper's and don't adjust during training.
# i quantize Lab color space by 262 (paper quantize by 313)
# i don't use data-dependent weight initialization.
###########################################################

if not(os.path.exists('/coco/data_preprocessed')):
    DP = data_preprocessor.DataPreprocessor()
    DP.run()


input_size = 20
lr = 0.0003    # 100 times of reference paper's initial learning rate
momentum = 0.9
decaying_factor = 0.00005
adam_beta1 = 0.9
adam_beta2 = 0.99
result_dir = './results'

max_epoch = 70
loss_sampling_step = 500

net = train.ColorizationNet(input_size, lr=lr, decaying_factor=decaying_factor,
                                adam_beta1=adam_beta1, adam_beta2=adam_beta2, result_dir=result_dir)
net.run(max_epoch, loss_sampling_step)
