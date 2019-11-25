import train.train_LRC
import train.train_BD
import train.train_PN
###########################################
# lr=0.0002, beta1=0.5, beta2=0.999, smoothing = 0.2
# LRC = train.train_LRC.CGAN(200)
# LRC.train(20, 1, 2, 2)
###########################################
###########################################
# lr=0.0002, beta1=0.5, beta2=0.999
# BD = train.train_BD.BD(100, lr = 0.0002)
# BD.train(20, 1, 2, 2)
###########################################
###########################################
# lr=0.0002, beta1=0.5, beta2=0.999, smoothing = 0.0
PN = train.train_PN.CGAN_PN(40, smoothing_factor=0.2)
PN.train(20, 1, 2, 2)