# 1.Overview
이 sub-Repository는 세 가지 이유로 만들어졌다.



1. 자동채색 프로젝트를 시작하고 최초에 [Consistent Comic Colorization with Pixel-wise Background Classification](https://nips2017creativity.github.io/doc/Consistent_Comic_Colorization.pdf)를 구현하려고 했지만, 구현과정에서 LRC 모듈을 PixcelCNN이 아닌 Pix2Pix로 구현하려다보니 전체적으로 내 구현이 원본논문의 구조와는 너무 달라져서 아예 다른 제목으로 리파지토리를 분리할 필요성이 있었다.



2. 자동채색을 구현하려는 과정에서의 최초의 시도와 고민이 본 README 문서에 담겨있기도 해서 추후 참고를 위해 남겨두는게 좋을 것 같았다.



3. 결과물이 만족스럽지 않은 실패한 실험이라도 실험데이터와 코드를 보존할 필요성이 있었다.



원본 논문에 Reference된 PixelCNN++와 Pixcolor를 어느정도 이해하고 난 뒤엔, 다시 원문의 구현에 충실하게 재구현에 도전해 볼 생각이다.



# 2.Implementation Log
## 2.1.Dataset

**2019-11-16**



현재 크롤러와 전처리기 모듈 추가완료




## 2.2.Low-resolution Colorizer

**2019-11-16**



Pixcolor: Pixel recursive colorization을 이해하기 위해선 영상 도메인에서의 Auto Regressive 모델들에 대해 알아야 한다. Auto Regressive 모델들은 대부분 단순 Image-to-Image CNN 모델들보다 Multimodality가 강하다는 점이 특징이다. 이를 위해 기본적으로 대부분의 모델들이 출력으로 Actual Value를 뱉는 것이 아니라 확률분포에서 샘플링하는 방법을 택하고 있다. 또한 Multimodality를 얻기위해 단순 가우시안 분포를 매핑하는 L2 Regression은 사용하지 않는다. 대신 카테고리컬 분포를 매핑하는 Softmax Classification이나 카테고리컬 분포가 이웃한 값간의 수학적 관계를 무시한다는 한계(반면 L2 regression은 이러한 수학적 관계를 잘 나타내지만 Multimodality를 모델링할 수 없다.)를 극복하기 위해, Logistic Mixture Model등의 모델을 사용하기도 한다. 



이렇게 모델링한 확률분포에서 샘플링을 하면 영상이 Multimodal해질까? 그렇지는 않은데, 모든 픽셀을 주변픽셀에 대한 고려없이 무작정 독립적으로 샘플링해버리면 그 결과물은 노이즈에 지나지 않으며 아마 Actual Value로 생성한 Single Mode 영상보다 퀄리티가 형편 없을 것이다. 따라서 우리는 각각의 픽셀이 종속성을 갖도록 영상을 생성하는 방법에 대해 생각해야 한다.



영상 도메인에서의 Auto Regressive 모델들의 기본가정은 영상공간에서의 픽셀들이 특정방향으로 Sequential하다는 것이다. 대표적인 Auto Regressive 모델인 PixelRNN을 살펴보면,



![img](./img/img3.PNG)



영상의 픽셀들이 좌상->우하 방향으로 Sequential하다는 것을 가정한 후, RNN계열의 Gated Unit들을 통해 좌상단에서부터 재귀적으로 픽셀을 샘플링해나간다. 이렇게 하면 분명히 퀄리티가 괜찮은 MultiMode 영상을 얻을 수 있을 것이다. 다만 RNN계열 모델 특성상 병렬처리를 지원하지 않아 엄청나게 느리다.



![img](./img/img4.PNG)



그래서 성능을 어느정도 내주고 학습속도를 얻은 PixelCNN같은 모델들이 등장하게 된다. 이 모델은 그림과 같이 생긴 Convolutional Filter를 이용해, 여러번 Convolve하여 마치 좌상단 피쳐맵으로부터 나온값이 중앙에 전달되고 이것들이 다시 Sequential하게 우하단으로 전달되는 모양새를 띈다.



이 논문에 사용된 Auto Regressive 모델은 PixelCNN++라는 모델인데 부끄러운 말이지만, 솔직히 그 구조가 구글논문답게 너무 기괴해서 단기간 내에 구현이 불가능하다는 결론에 이르렀다. 결국 시간이 촉박해 일단 다른 모듈과 같은 CGAN으로 구현하였는데 성능이 괜찮다. 원저자는 세 모델 모두Adverserial Loss를 사용하지 않았다는데 나는 Adverserial Loss를 사용해봤더니 성능이 상당히 괜찮았다. 그런데 Low-resolution Colorizer의 성능이 너무 좋은게 밑에 Background Detector를 학습시키는데 문제를 일으켰다.



## 2.3.Background Detector

**2019-11-16**



위에서 구현한 CGAN 기반 Low-resolution Colorizer가 너무 잘 Fitting되어서 사실상 Ground Truth와 유사한 값만 뱉는 상황이 왔다.(...) 저자의 의도대로 Background Detector를 학습시키려면, Low-resolution Colorizer가 전경만 잘 예측하고 후경은 잘 예측하지 못해야 한다. 그런데 후경도 너무 잘 예측하니까 학습이 전혀되질 않는다. (그냥 모든 영역을 foreground로 볼려함)



해결책으로 다음과 같은 시도를 해보았는데,



1. 좀 더 앞선 Epoch에서 학습된 Low-resolution colorizer를 사용해보았지만 이건 전경도 예측을 잘 못하는 문제가 있었다.



2. Low-resolution colorizer에 노이즈를 줘봤는데 뭔가 나아진 것 같은 기분이 드는 것도 같은데 솔직히 눈에 띄는 개선이 없었다.



그렇다고 300장 남짓한 Test set만 가지고 학습을 시키는 건 의미가 없다. 그래서 그냥 아예 역발상으로 BD를 생략하고 바로 Polishing Network로 연결하는 방법을 시도해 보려고 한다. 이게 안되면 그냥 데이터를 더 크롤링해오는 수 밖에 방법이 없다. 다음부터 이런 자기 모델의 잠재적인 약점(후경을 잘 예측하지 못하는 것)을 이용하는 구조를 설계할 때는 학습용 데이터셋을 좀 더 세분화할 필요가 있겠다.



BD를 없앤다는 것은 결국 Pix2Pix를 2개 붙여놓은 것과 거의 다름이 없기 때문에, 이 구조의 이름을 Naive 2-Step CGAN로 이름 붙였다.



## 2.4.Polishing Network

**2019-11-16**



3번, 5번을 생략하고 1, 2, 4번만으로 예측하는 모델을 짜도록 한다. Pix2Pix의 구조를 그대로 따왔고, 디테일한 특징을 살리기위해 Adverderial Loss도 사용하여 구현하였다.



# 3.Result


![img](./img/result.png)



원문의 구현과 달리, Naive하게 2-Step의 Pix2Pix로 구현한 결과 색이 마구 번지고 섞여있어 만족스럽지 않은 결과를 보여준다.
