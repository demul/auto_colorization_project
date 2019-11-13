auto_colorization_project

# 1.Paper Study


## 1.1.Consistent Comic Colorization with Pixel-wise Background Classification([https://nips2017creativity.github.io/doc/Consistent_Comic_Colorization.pdf])



[img]



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
