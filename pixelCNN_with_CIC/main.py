import data_preprocessor
import train
import os

###########################################################
# differences with reference paper(PixColor) :
# my dataset is 'MS-COCO'
#
# i use Lab color space instead of YCbCr because Lab is closer to Human Vision System than YCbCr.
# i quantize Lab color space by 262
#
# i don't use pre-trained conditioning network.
#
# i use joint distribution of a, b color channel
# because dependencies between the color channels of a pixel is are likely to be relatively simple and
# do not require a deep network to model.
# so this model predict both channel jointly than predict each channel recursively, sequentially.
#
# the architecture can be somewhat different with original's
# because the paper omits about details of model's architecture
# (ex : adaptation network's, inexplicable input channel dimension of PixelCNN's, etc...).
# you can see my architecture on [https://github.com/demul/auto_colorization_project/blob/master/pixcolor/README.md]
###########################################################

if not(os.path.exists('/coco/data_preprocessed_PixColor_class')
       and os.path.exists('/coco/data_preprocessed_PixColor_ab')):
    DP = data_preprocessor.DataPreprocessor()
    DP.run()


input_size = 8
lr = 0.0003
momentum = 0.9
decaying_factor = 0.00005
adam_beta1 = 0.9
adam_beta2 = 0.9
result_dir = './results'

max_epoch = 10
loss_sampling_step = 500

net = train.ColorizationNet(input_size, lr=lr, decaying_factor=decaying_factor,
                            adam_beta1=adam_beta1, adam_beta2=adam_beta2, result_dir=result_dir)
net.run(max_epoch, loss_sampling_step)
