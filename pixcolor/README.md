# 1.Overview
구현과정에서의 특징과 원본 논문과의 차이점을 기술한다.



# 2.Detail


## 2.1.Color Space


Color Space로, YCbCr대신 사람의 시각인지에 가장 가까운 Lab Color Space를 사용했다.


## 2.2.Distribution 


![img](./img/pixcolor4)



PixelCNN++에 따르면, 영상에서 채널간의 Dependency가 그렇게 복잡하지 않으므로, a, b 두 채널을 각각 Sequential하게 예측하는 대신, Joint Distribution을 한꺼번에 예측하도록 했다. a, b의 순서쌍은 총 262개로 Quantization했다.

## 2.3.Architecture


![img](./img/pixcolor5)



신경망의 구조에 대해서, 논문에 정확히 서술되어 있지 않거나 애매하게 설명된 부분들이 있다. Adaptation Network의 구조가 설명되어 있지 않고, Pixel CNN Colorization Network의 Input이 28 x 28 x 2인 것도 이상하다.  Conditioning Feature Map과 자신의 Output을 Recursive하게 받을려면 채널이 2보다 커야 하기 때문이다.



그런데 저자가 메일도 안 받고 제대로 된 공개구현체도 없어서 애매한 부분들은 그냥 직접 설계하기로 했다. 다만 직접 설계하되, 최대한 논문에 나와있는 단서들과 설계철학을 반영하여 설계했다.



![img](./img/pixcolor6)



먼저 Adqptation Network는 간단하게 3x3 Convolution Layer 3개를 쌓아서 구성했다. Output의 차원은 28 x 28 x 64이고, 이것이 PixelCNN의 Output에서 샘플링한 28 x 28 x 2의 데이터와 Concat되어, 28 x 28 x 66의 데이터가 되어 PixelCNN의 Input으로 들어간다.



내가 재설계한 Architecture를 표로 나타내면 다음과 같다.


![img](./img/'my architecture.png')
