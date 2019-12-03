from data_crawling import Crawler_
from data_preprocessing import  Preprossesor
import train.train_LRC
import train.train_PN

################ Crawl test data #################
# C = Crawler_(242, 342, 'test')
# C.run()


############# Preprocess test data ###############
# P = Preprossesor('test')
# P.preprocess_save()


######### Make Low-resolution test data ##########
# data_maker = train.train_LRC.CGAN(20, input_dir = "test") #반드시 LRC가 학습된 뒤에 실행
# data_maker.make_data_for_BD(2,2, start=0, end = 1000)


####### Make High-resolution final result ########
# lr=0.0002, beta1=0.5, beta2=0.999, smoothing = 0.0
PN = train.train_PN.CGAN_PN(40, smoothing_factor=0.2, input_dir = "test")
PN.save_final_result(2,2, start=0, end = 1000)

# PN = train.train_PN.CGAN_PN(40, smoothing_factor=0.2, input_dir = "yumi_cell")
# PN.save_final_result(2,2, start=0, end = 7380)