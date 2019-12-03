![img](./img/result.png)
# 1.Overview
이 sub-Repository는 두 가지 목적으로 만들어 졌다.



1. [Consistent Comic Colorization with Pixel-wise Background Classification](https://nips2017creativity.github.io/doc/Consistent_Comic_Colorization.pdf)을 구현하려는 시도중, LRC 모듈을 PixcelCNN이 아닌 2단계의 Pix2Pix로 구현하려다가 결과적으로 좋지 못한 성능을 내게 되어, 전체적으로 코드를 갈아엎기로 결정한 뒤, 실험데이터와 코드를 분리하여 보존하기 위한 목적.




2. 자동채색을 구현하려는 과정에서의 최초의 시도와 고민, 공부가 본 README 문서에 담겨있기도 하기에 추후 참고를 위한 목적.


이다.



내 제멋대로인데다 형편없는 구현이 논문 저자에게 너무 미안해서, 인간된 도리로 리파지토리 이름을 논문명과 다르게 설정하였다. 우연히라도 이 sub-Repository에 오게 된 사람은 구현은 절대 참고하지 말고(원문 구현과 다른 요소가 너무 많다.) 논문을 정리해놓은 [부분](# 2.consistent-comic-colorization-with-pixel-wise-background-classification)만 참고하길 바란다. 논문이해는 꽤 잘한 것 같다고 저자가 칭찬도 해줬으니 논문 정리 부분은 믿어도 좋다.



이 논문에 Refference된 PixelCNN++와 Pixcolor를 어느정도 이해하고 난 뒤엔, 다시 원문의 구현에 충실하게 재구현에 도전해 볼 생각이다.



# 2.[Consistent Comic Colorization with Pixel-wise Background Classification](https://nips2017creativity.github.io/doc/Consistent_Comic_Colorization.pdf)



![img](./img/img1.JPG)


## 2.1.Idea
이 논문의 주요한 Contribution은 기존에 제안되었던 Auto-colorization기법들이 Background-consistent한 결과물을 내놓지 못하던 문제를 Background Detector를 도입함으로서 해결하였다는 것이다.
이 논문에 제안된 모델은



1. 실질적인 Outline Colorization을 담당하는 **Low-resolution Colorizer**, 


2. 전경과 후경을 분류하는 **Background Detector**, 


3. 최초 Input인 Outline, Low-resolution Colorizer의 Output인 Low-resolution Image, Background Detector의 Output인 Background-Foreground Segment를 받아 Background-consistent하게 Resolution을 복구하는 **Polishing Network**로 나뉜다.


이 논문의 핵심 아이디어인 Background Detector는 전, 후경을 직접 Labeling하는 지도학습 방법대신, 독특한 비지도학습 방법으로 학습된다. 


마스크를 하나 만든 뒤, 전경으로 분류된 픽셀은 Low-resolution Image의 값으로, 후경으로 분류된 픽셀은 Ground-Truth에서 해당 Index에 해당하는 픽셀의 값들을 얻은 뒤 Mean한 값 하나로, 즉 단색으로 덮어버린다. 이렇게 만든 이미지와 Ground-Truth의 L1-loss를 최소화하게 학습시킨다. 



이게 되는 이유는 캐릭터나 말풍선의 색은 다수의 데이터에 걸쳐 유사한 패턴으로 반복되는 반면, 후경의 색은 일반적으로 뚜렷한 패턴없이 정해지기 때문이다. 따라서 Low-resolution Colorizer는 전경에 해당하는 부분(말풍선, 캐릭터 등)은 잘 복원하는 반면 후경에 해당하는 부분은 잘 복원하지 못 할것이다. 


또한 일반적으로 일반적으로 후경은 Consistent하기 때문에 해당 부분 픽셀들을 Mean해서 단색으로 덮어도 Ground-Truth와 차이가 크지 않을 것이다. 따라서 Low-resolution Colorizer에 의해 복원된 후경보다 Ground-Truth로부터 Mean해서 얻은 단색후경이 Ground-Truth에 더 가까울 것이다. 



"뭐하러 Mean한 단색으로 덮는가, 그냥 Ground Truth를 그대로 쓰면 안되냐?"라는 의문이 들 수 있지만 조금만 생각해보면 이게 불가능하다는 것을 알 수 있다. 


Ground Truth를 그대로 쓰면 그냥 이미지 전체를 후경으로 취급해버리는게 L1 Loss가 가장 작게 나오기 때문이다. 일반적으로 후경은 Consistent하고 캐릭터, 말풍선은 후경과 뚜렷하게 구분되게 그려진다는 만화의 Domain적 특성을 잘 활용한 상당히 깜찍한 아이디어라고 할 수 있다. 다만 이러한 특성이 



1. **전경과 후경이 뚜렷히 구분되지 않게 그려지는 화풍을 가진 만화**(예를 들어 흰 후경이 잦은 만화를 학습데이터로 쓴다면 아마 말풍선과 후경을 잘 분리하지 못하게 학습될 것이다.),


![img](./img/img2.JPG)


2. **배경이 단색이 아닌 복잡한 패턴을 가져 Mean해버리기에 무리가 있는 만화**(애초에 이 모델은 후경이 어느정도 Consistent하다는 가정에서 짜여진 모델이다.)



들에 대해 한계를 가지게 될 것 같다.




## 2.2.Detail
### 2.2.1.Dataset
유미의 세포들이라는 만화의 첫 화부터 238화까지, 총 7394개 이미지를 256x256으로 resize해서 사용했다고 한다. 데이터를 대충 훑어보니 대부분이 컷 분할이 깔끔하고 종횡비의 차가 크지 않은 컷이라 데이터셋으로 쓰기 좋아보였다. 생각보다 데이터가 깔끔해서 아마 전처리보다는 크롤러 만들어서 긁어오는데 더 많은 시간을 소요할 것으로 보인다.




### 2.2.2.Low-resolution Colorizer
 기본적인 구조는 Pixcolor: Pixel recursive colorization([https://arxiv.org/abs/1705.07208])의 것을 따르고 있으며, 전이학습을 하지 않는 점, 적은 Dataset에 대해 더 나은 성능을 얻기위해 Logistic Mixture Model([https://arxiv.org/abs/1701.05517])을 사용했다는 점이 차이점이다. Canny-edge와 원래 검게 칠해진 부분을 더해 얻은 Outline을 Input으로 하고 Ground-truth를 32x32까지 Downsample한 영상을 Output으로 하는 전형적인 Pix2PixCNN으로 보인다.(**라고 생각했던 것은 내 큰 착각이었다.**)




### 2.2.3.Background Detector
기본적인 구조는 Image-to-Image Translation with Conditional Adversarial Networks([https://arxiv.org/abs/1611.07004])의 것을 따르고 있으며, 최종단의 Binary한 Output을 Gumbel-Softmax로 얻는다.(https://arxiv.org/abs/1611.01144) 이후 위에 언급했듯, 전경으로 분류된 부분엔 Low-resolution Colorizer의 값을 곱하고, 후경으로 분류된 부분엔 같은 Index를 가진 Ground-truth 값들의 평균을 곱한다. 이 둘을 합해 얻은 이미지와 Ground-truth간의 L1 Loss를 Minimize하게 학습시킨다. 




### 2.2.4.Polishing Network
기본적인 구조는 Image-to-Image Translation with Conditional Adversarial Networks([https://arxiv.org/abs/1611.07004])의 것을 따르고 있으며, 


1. 최초 Input인 Outline(1 channel), 


2. Low-resolution Colorizer의 Output인 Low-resolution Image(3 channel),


3. Background Detector의 Output인 Background-Foreground Segment(각각 반전하여 하나씩 2 channel)


4. Mask(1 channel)


5. Low-resolution Image의 후경 픽셀들의 값의 평균(3 channel)


총 10 channel의 Image를 Input으로 받아 Background-consistent하게 Resolution을 복구한다. 



모델이 1번 Low-resolution Image에 과의존하는 것을 막기위해 Low-resolution Image에서 후경으로 분류된 부분 중 일부를 Random하게 Mask하는데 이때 Masking하는 값은 앞서 숱하게 사용한 후경 픽셀들의 값의 평균이다. 

4번 Mask는 Random Noise 줄 때 사용한 Noise의 마스크이다.


5번 Low-resolution Image의 후경 픽셀들의 값의 평균(3 channel)은 Background로 분류된 픽셀들에만 Masking한 형태로 사용한다.




# 3.Implementation
### 3.1.Dataset

**2019-11-16**



현재 크롤러와 전처리기 모듈 추가완료




### 3.2.Low-resolution Colorizer

**2019-11-16**



Pixcolor: Pixel recursive colorization을 이해하기 위해선 영상 도메인에서의 Auto Regressive 모델들에 대해 알아야 한다. Auto Regressive 모델들은 대부분 단순 Image-to-Image CNN 모델들보다 Multimodality가 강하다는 점이 특징이다. 이를 위해 기본적으로 대부분의 모델들이 출력으로 Actual Value를 뱉는 것이 아니라 확률분포에서 샘플링하는 방법을 택하고 있다. 또한 Multimodality를 얻기위해 단순 가우시안 분포를 매핑하는 L2 Regression은 사용하지 않는다. 대신 카테고리컬 분포를 매핑하는 Softmax Classification이나 카테고리컬 분포가 이웃한 값간의 수학적 관계를 무시한다는 한계(반면 L2 regression은 이러한 수학적 관계를 잘 나타내지만 Multimodality를 모델링할 수 없다.)를 극복하기 위해, Logistic Mixture Model등의 모델을 사용하기도 한다. 



이렇게 모델링한 확률분포에서 샘플링을 하면 영상이 Multimodal해질까? 그렇지는 않은데, 모든 픽셀을 주변픽셀에 대한 고려없이 무작정 독립적으로 샘플링해버리면 그 결과물은 노이즈에 지나지 않으며 아마 Actual Value로 생성한 Single Mode 영상보다 퀄리티가 형편 없을 것이다. 따라서 우리는 각각의 픽셀이 종속성을 갖도록 영상을 생성하는 방법에 대해 생각해야 한다.



영상 도메인에서의 Auto Regressive 모델들의 기본가정은 영상공간에서의 픽셀들이 특정방향으로 Sequential하다는 것이다. 대표적인 Auto Regressive 모델인 PixelRNN을 살펴보면,



![img](./img/img3.PNG)



영상의 픽셀들이 좌상->우하 방향으로 Sequential하다는 것을 가정한 후, RNN계열의 Gated Unit들을 통해 좌상단에서부터 재귀적으로 픽셀을 샘플링해나간다. 이렇게 하면 분명히 퀄리티가 괜찮은 MultiMode 영상을 얻을 수 있을 것이다. 다만 RNN계열 모델 특성상 병렬처리를 지원하지 않아 엄청나게 느리다.



![img](./img/img4.PNG)



그래서 성능을 어느정도 내주고 학습속도를 얻은 PixelCNN같은 모델들이 등장하게 된다. 이 모델은 그림과 같이 생긴 Convolutional Filter를 이용해, 여러번 Convolve하여 마치 좌상단 피쳐맵으로부터 나온값이 중앙에 전달되고 이것들이 다시 Sequential하게 우하단으로 전달되는 모양새를 띈다.



이 논문에 사용된 Auto Regressive 모델은 PixelCNN++라는 모델인데 부끄러운 말이지만, 솔직히 그 구조가 구글논문답게 너무 기괴해서 단기간 내에 구현이 불가능하다는 결론에 이르렀다. 결국 시간이 촉박해 일단 다른 모듈과 같은 CGAN으로 구현하였는데 성능이 괜찮다. 원저자는 세 모델 모두Adverserial Loss를 사용하지 않았다는데 나는 Adverserial Loss를 사용해봤더니 성능이 상당히 괜찮았다. 그런데 Low-resolution Colorizer의 성능이 너무 좋은게 밑에 Background Detector를 학습시키는데 문제를 일으켰다.



### 3.3.Background Detector

**2019-11-16**



위에서 구현한 CGAN 기반 Low-resolution Colorizer가 너무 잘 Fitting되어서 사실상 Ground Truth와 유사한 값만 뱉는 상황이 왔다.(...) 저자의 의도대로 Background Detector를 학습시키려면, Low-resolution Colorizer가 전경만 잘 예측하고 후경은 잘 예측하지 못해야 한다. 그런데 후경도 너무 잘 예측하니까 학습이 전혀되질 않는다. (그냥 모든 영역을 foreground로 볼려함)



해결책으로 다음과 같은 시도를 해보았는데,



1. 좀 더 앞선 Epoch에서 학습된 Low-resolution colorizer를 사용해보았지만 이건 전경도 예측을 잘 못하는 문제가 있었다.



2. Low-resolution colorizer에 노이즈를 줘봤는데 뭔가 나아진 것 같은 기분이 드는 것도 같은데 솔직히 눈에 띄는 개선이 없었다.



그렇다고 300장 남짓한 Test set만 가지고 학습을 시키는 건 의미가 없다. 그래서 그냥 아예 역발상으로 BD를 생략하고 바로 Polishing Network로 연결하는 방법을 시도해 보려고 한다. 이게 안되면 그냥 데이터를 더 크롤링해오는 수 밖에 방법이 없다. 다음부터 이런 자기 모델의 잠재적인 약점(후경을 잘 예측하지 못하는 것)을 이용하는 구조를 설계할 때는 학습용 데이터셋을 좀 더 세분화할 필요가 있겠다.




### 3.4.Polishing Network

**2019-11-16**



3번, 5번을 생략하고 1, 2, 4번만으로 예측하는 모델을 짜도록 한다. Pix2Pix의 구조를 그대로 따왔고, 디테일한 특징을 살리기위해 Adverderial Loss도 사용하여 구현하였다.
