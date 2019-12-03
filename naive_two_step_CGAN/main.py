import train.train_LRC
import train.train_BD
import train.train_PN

###########################################
# lr=0.0002, beta1=0.5, beta2=0.999, smoothing = 0.2
# LRC = train.train_LRC.CGAN(40, lr=0.0002, smoothing_factor=0.2)
# LRC.train(500, 20, 2, 2)
###########################################
###########################################
# lr=0.0002, beta1=0.5, beta2=0.999
# BD = train.train_BD.BD(100, lr=0.002, input_dir = "more_data")
# BD.train(200, 1, 2, 2, train_size=5800, test_size=100)
###########################################
###########################################
# lr=0.0002, beta1=0.5, beta2=0.999, smoothing = 0.0
PN = train.train_PN.CGAN_PN(40, lr=0.0002, smoothing_factor=0.2, input_dir = "more_data")
PN.train(200, 1, 2, 2, train_size=5800, test_size=100)