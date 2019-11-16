auto_colorization_project

# 1.Paper Study


## 1.1.[Consistent Comic Colorization with Pixel-wise Background Classification](https://nips2017creativity.github.io/doc/Consistent_Comic_Colorization.pdf)



[img]


### 1.1.1.Idea
이 논문의 주요한 Contribution은 기존에 제안되었던 Auto-colorization기법들이 Background-consistent한 결과물을 내놓지 못하던 문제를 Background Detector를 도입함으로서 해결하였다는 것이다.
이 논문에 제안된 모델은



1. 실질적인 Outline Colorization을 담당하는 **Low-resolution Colorizer**, 


2. 전경과 후경을 분류하는 **Background Detector**, 


3. 최초 Input인 Outline, Low-resolution Colorizer의 Output인 Low-resolution Image, Background Detector의 Output인 Background-Foreground Segment를 받아 Background-consistent하게 Resolution을 복구하는 **Polishing Network**로 나뉜다.



학습단계에서 단 모듈은 독립적으로 학습된다. 이 논문의 핵심 아이디어인 Background Detector는 전, 후경을 직접 Labeling하는 지도학습 방법대신, 독특한 비지도학습 방법으로 학습된다. 


마스크를 하나 만든 뒤, 전경으로 분류된 픽셀은 Low-resolution Image의 값으로, 후경으로 분류된 픽셀은 Ground-Truth에서 해당 Index에 해당하는 픽셀의 값들을 얻은 뒤 Mean한 값 하나로, 즉 단색으로 덮어버린다. 이렇게 만든 이미지와 Ground-Truth의 L1-loss를 최소화하게 학습시킨다. 



이게 되는 이유는 캐릭터나 말풍선의 색은 다수의 데이터에 걸쳐 유사한 패턴으로 반복되는 반면, 후경의 색은 일반적으로 뚜렷한 패턴없이 정해지기 때문이다. 따라서 Low-resolution Colorizer는 전경에 해당하는 부분(말풍선, 캐릭터 등)은 잘 복원하는 반면 후경에 해당하는 부분은 잘 복원하지 못 할것이다. 


또한 일반적으로 일반적으로 후경은 Consistent하기 때문에 해당 부분 픽셀들을 Mean해서 단색으로 덮어도 Ground-Truth와 차이가 크지 않을 것이다. 따라서 Low-resolution Colorizer에 의해 복원된 후경보다 Ground-Truth로부터 Mean해서 얻은 단색후경이 Ground-Truth에 더 가까울 것이다. 



"뭐하러 Mean한 단색으로 덮는가, 그냥 Ground Truth를 그대로 쓰면 안되냐?"라는 의문이 들 수 있지만 조금만 생각해보면 이게 불가능하다는 것을 알 수 있다. 


Ground Truth를 그대로 쓰면 그냥 이미지 전체를 후경으로 취급해버리는게 L1 Loss가 가장 작게 나오기 때문이다. 일반적으로 후경은 Consistent하고 캐릭터, 말풍선은 후경과 뚜렷하게 구분되게 그려진다는 만화의 Domain적 특성을 잘 활용한 상당히 깜찍한 아이디어라고 할 수 있다. 다만 이러한 특성이 



1. **전경과 후경이 뚜렷히 구분되지 않게 그려지는 화풍을 가진 만화**(예를 들어 흰 후경이 잦은 만화를 학습데이터로 쓴다면 아마 말풍선과 후경을 잘 분리하지 못하게 학습될 것이다.),



2. **단색이 아닌 복잡한 패턴을 가져 Mean해버리기에 무리가 있는 만화**(애초에 이 모델은 후경이 어느정도 Consistent하다는 가정에서 짜여진 모델이다.)



들에 대해 한계를 가지게 될 것 같다.




### 1.1.2.Detail
#### 1.1.2.1.Dataset
유미의 세포들이라는 만화의 첫 화부터 238화까지, 총 7394개 이미지를 256x256으로 resize해서 사용했다고 한다. 데이터를 대충 훑어보니 대부분이 컷 분할이 깔끔하고 종횡비의 차가 크지 않은 컷이라 데이터셋으로 쓰기 좋아보였다. 생각보다 데이터가 깔끔해서 아마 전처리보다는 크롤러 만들어서 긁어오는데 더 많은 시간을 소요할 것으로 보인다.



**2019-11-16**



현재 크롤러와 전처리기 모듈 추가완료



#### 1.1.2.2.Low-resolution Colorizer
 기본적인 구조는 Pixcolor: Pixel recursive colorization([https://arxiv.org/abs/1705.07208])의 것을 따르고 있으며, 전이학습을 하지 않는 점, 적은 Dataset에 대해 더 나은 성능을 얻기위해 Logistic Mixture Model([https://arxiv.org/abs/1701.05517])을 사용했다는 점이 차이점이다. Canny-edge와 원래 검게 칠해진 부분을 더해 얻은 Outline을 Input으로 하고 Ground-truth를 32x32까지 Downsample한 영상을 Output으로 하는 전형적인 Pix2PixCNN으로 보인다.



#### 1.1.2.3.Background Detector
기본적인 구조는 Image-to-Image Translation with Conditional Adversarial Networks([https://arxiv.org/abs/1611.07004])의 것을 따르고 있으며, 최종단의 Binary한 Output을 Gumbel-Softmax로 얻는다.(https://arxiv.org/abs/1611.01144) 이후 위에 언급했듯, 전경으로 분류된 부분엔 Low-resolution Colorizer의 값을 곱하고, 후경으로 분류된 부분엔 같은 Index를 가진 Ground-truth 값들의 평균을 곱한다. 이 둘을 합해 얻은 이미지와 Ground-truth간의 L1 Loss를 Minimize하게 학습시킨다. 



#### 1.1.2.4.Polishing Network
기본적인 구조는 Image-to-Image Translation with Conditional Adversarial Networks([https://arxiv.org/abs/1611.07004])의 것을 따르고 있으며, 


1. 최초 Input인 Outline(1 channel), 


2. Low-resolution Colorizer의 Output인 Low-resolution Image(3 channel),


3. Background Detector의 Output인 Background-Foreground Segment(각각 반전하여 하나씩 2 channel)


4. Mask(1 channel)


5. Low-resolution Image의 후경 픽셀들의 값의 평균(3 channel)


총 10 channel의 Image를 Input으로 받아 Background-consistent하게 Resolution을 복구한다. 



모델이 1번 Low-resolution Image에 과의존하는 것을 막기위해 Low-resolution Image에서 후경으로 분류된 부분 중 일부를 Random하게 Mask하는데 이때 Masking하는 값은 앞서 숱하게 사용한 후경 픽셀들의 값의 평균이다. 

4번 Mask가 Random Noise 줄 때 사용한 Noise의 마스크인건지... 다소 모호하게 논문에 적혀있었으므로 저자에게 직접 질문메일을 보냈다. 이를 반영해 본문서를 수정해 나갈 예정이다.


5번 Low-resolution Image의 후경 픽셀들의 값의 평균(3 channel)은 Background로 분류된 픽셀들에만 Masking한 형태로 사용한다.


