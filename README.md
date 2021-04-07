# Segnet

## 1. 개요

---

![images/Untitled%2034.png](images/Untitled%2034.png)

그림 8: [https://mi.eng.cam.ac.uk/projects/segnet/](https://mi.eng.cam.ac.uk/projects/segnet/)

**Segmentation**이란 *그림 8*과 같이 이미지 상에 있는 객체들을 픽셀 단위로 구분짓는것을 말한다. Segmentation에는 두개의 종류가 있는데, 같은 종류의 객체를 하나의 그룹으로 묶는 **Semantic Segmentation**(*그림 9*에서 왼쪽 사진)과 같은 종류의 객체일지라도 서로 다른 그룹으로 구분하는 **Instance Segmentation**(*그림 9*에서 오른쪽 사진)이 있다.

![images/Untitled%2035.png](images/Untitled%2035.png)

그림 9: [https://www.researchgate.net/profile/Vinorth-Varatharasan/publication/339616270/figure/fig1/AS:864483257896960@1583120278427/Semantic-segmentation-left-and-Instance-segmentation-right-8.ppm](https://www.researchgate.net/profile/Vinorth-Varatharasan/publication/339616270/figure/fig1/AS:864483257896960@1583120278427/Semantic-segmentation-left-and-Instance-segmentation-right-8.ppm)

*그림 10*을 보면 Segmentation에서 output은 객체마다 할당된 인덱스가 픽셀 단위로 계산되어 있는데, input의 dimension이 W*H*3 이라면 output의 dimension은 W*H*1 이다.

![images/Untitled%2036.png](images/Untitled%2036.png)

그림 10: [https://i.imgur.com/Sp5l9P9.png](https://i.imgur.com/Sp5l9P9.png)

Segmentation을 위한 Deep learning 기반 네트워크는 다양하게 존재한다. 그 중 가장 기본적인 구조를 가지고 있으면서 실시간성이 어느정도 보장되는 **SegNet** 에 대해 분석해보고자 한다.

SegNet은 일반적인 CNN 기반의 **Encoder-Decoder Architecture**를 가지고 있다. Segmentation은 위에서도 볼 수 있듯이 input의 Width와 output의 Width가 같은데, 이를 위해 *그림 11*처럼 네트워크에서 Pooling을 사용하지 않고 항상 동일한 width를 가진 feature map이 나오도록 한다면 그만큼 연산량도 많아지고 비효율적인 형태가 될 것이다. 따락서 최근의 Segmentation 모델들을 *그림 12*과 같이 일반 CNN과 같이 연산을 진행하여 feature map의 크기를 줄인 후(**downsampling**) 다시 feature map의 크기를 원본 이미지와 같이 키우는(**upsampling**) 형태로 구성이 되어있다.

![images/Untitled%2037.png](images/Untitled%2037.png)

그림 11: [https://tariq-hasan.github.io/assets/images/semantic_segmentation_fully_convolutional_vanilla.png](https://tariq-hasan.github.io/assets/images/semantic_segmentation_fully_convolutional_vanilla.png)

![images/Untitled%2038.png](images/Untitled%2038.png)

그림 12: [https://tariq-hasan.github.io/assets/images/semantic_segmentation_fully_convolutional_sampling.png](https://tariq-hasan.github.io/assets/images/semantic_segmentation_fully_convolutional_sampling.png)

뒤에서는 Pytorch 기반으로 구현된 SegNet 을 분석해 볼 것이다.

## 2. 구성

---

내가 사용한 프로젝트 코드의 구성을 알아보고, SegNet 모델 구성에 대해 알아보았다.

### 2.1 소스코드 구성

소스코드 구성은 아래와 같은 트리구조로 되어있다.

- **src**: 모든 소스코드가 담겨있는폴더
    - **train.py**: 모델 학습(train) 로직 구현
    - **inference.py**: 모델 평가(evaluation) 로직 구현
    - **models.py**: 딥러닝 네트워크 모델 구현
    - **dataset.py**: 데이터셋 처리 함수, loader 구현

### 2.2 모델 구성

SegNet은 *그림 13*과 같이 CNN 기반의 Encoder-Decoder Architecture를 가지고 있다. Encoder, Decoder 특징을 정리해 보았다.

![images/Untitled%2082.png](images/Untitled%2082.png)

그림 13: [https://arxiv.org/pdf/1511.00561.pdf](https://arxiv.org/pdf/1511.00561.pdf)

- **Encoder**
    - 기존 VGG16 네트워크에서 FC 층을 제외한 나머지 Convolution층 13개를 그대로 가져온 형태를 보이고 있다.
    - 2x2의 max pooling 을 사용하여 feature map의 가로, 세로 크기를 절반으로 줄이고 있다.
- **Decoder**
    - Encoder와 대칭적인 형태를 가지고 있다.
    - 중간에 upsampling을 사용하여 feature map의 가로, 세로 크기를 두배로 늘리고 있으며 이때 *그림 14*와 같이 max pooling indecies 를 사용한다 (상응하는 max pooling layer에서 max 값이 있었던 위치에 값을 넣고 나머지는 0으로 채우는 형태)
    - 마지막에 Softmax 층을 통해 각 pixel의 class를 유추한다

![images/Untitled%2083.png](images/Untitled%2083.png)

그림 14: [https://arxiv.org/pdf/1511.00561.pdf](https://arxiv.org/pdf/1511.00561.pdf)

## 3 활용 및 결과

---

![images/Untitled%2084.png](images/Untitled%2084.png)

그림 15: [https://arxiv.org/pdf/1511.00561.pdf](https://arxiv.org/pdf/1511.00561.pdf)

SegNet은 Segmentation을 하는 다른 모델인 U-Net, DeconvNet과 비교했을 때 비교적 간단한 아키텍처로 되어있다. 그에 따라 더 적은 weight들을 사용하고 inference 속도도 빠른 형태를 보여준다. 따라서 Yolo와 마찬가지로 다른 모델에 비해 실시간성이 뛰어나 자율주행에서 사용하기에 적합하다. 자율주행을 하기 위해서는 주행 공간에 있는 객체들을 인식해야 하고 차량이 갈 수 있는 공간과 갈 수 없는 공간을 구분해야 한다. 이러한 정보를 얻는데 SegNet 모델을 활용할 수 있을 것이다.