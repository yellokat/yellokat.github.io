---
math: true
layout: post
title: "[추천시스템 논문 리뷰] CL4CTR: A Contrastive Learning Framework for CTR Prediction"
date: 2024-01-05 19:32 +0900
categories:
  - Papers
tags:
  - Recommender Systems
  - Multi-Modal Recommendation
  - Machine Learning
  - Deep Learning
---
# 1. Introduction

CTR 예측 태스크는 크게 두 가지로 나눌 수 있다.

- Traditional Method: Logistic Regression 등 및 Factorization Machine 등.
- 딥러닝 기반 기법: DeepFM, NeuMF 등

현재 SOTA인 모든 방법들은 backpropagation과 SGD를 통해 학습한다. 이 과정에서 문제가 생기는데, feature별로 학습에 참여하는 횟수가 다르다는 것이다. 각 feature가 학습에 참여하는 빈도는 long-tail 분포를 그린다. 따라서 **sparse하게 등장하는 feature의 representation embedding은 제대로 학습되기 어렵다**. 

이를 해결하기 위한 기존 방법론으로 각 feature에 대한 embedding update시 가중치 parameter 등을 더하는 방법이 제시되었으나, 이는 결국 파라미터 수를 늘리고 추론 속도를 저하시킨다. 이 논문에서는 가중치를 사용하지 않는 다른 방법으로 sparse feature의 representation learning 성능 향상을 노린다. 또한 이 논문에서 제시하는 방법은 Model-Agnostic하다.

CL4CTR는 3가지 모듈로 구성된다.

- **CTR Prediction module:**
- **Contrastive Module:**
- **Alignment & Uniformity Constraints.**

# 2. The CL4CTR Framework

## 2-1. **CTR Prediction**

추천 문제를 이진분류 과제로 cast 하여 푼다. 이 모듈에서 **CTR Loss**가 생성된다.

## 2-2. **Contrastive Module**

Data Augmentation을 통해 두 개의 Synthetic Data를 생성하고, 두 representation의 차이가 적어지도록 학습한다. 이 모듈에서 **Contrastive Loss**가 생성된다. 다음의 세 단계로 이루어진다.

- Data Augmentation:
- Feature Interaction Encoder:
- Contrastive Loss Function

## 2-3. **Feature Alignment & Field Uniformity**

타 분야(CV, NLP 등)의 Contrastive Learning에서는 Alignment & Uniformity의 개념이 자주 사용된다. 비슷한 것의 Representation은 가깝게, 다른 것의 Representation은 멀게 학습시키는 것이다.

이 논문에서는 이것을 추천시스템의 도메인에 맞게 해석하였다. 

- **같은 Field의 Feature표현은 서로 가까워지도록** 학습시키고,
- 서로 **다른 Field의 Feature 표션은 멀어지도록** 학습하는 것이다.

이것을 Feature Alignment & Field Uniformity라고 부르기로 한다.

### **Feature Alignment의 Loss function**

각 필드 $F$에 대해서, 그 필드 내의 모든 두 embedding pair 사이의 거리를 가깝게 하고 있다.

$$
\mathcal{L}_a=\sum^{F}_{f=1}\sum_{e_i, e_j\in E_f}||e_i-e_j||^2_2
$$

### **Field Uniformity의 Loss function**

어떤 필드 속의 feature embedding 하나에 대해, 그 필드만 빼고 다른 모든 필드의 모든 feature embedding과의 similarity를 최소화하고 있다.

$$
\mathcal{L}_u=\sum_{e_i\in E_f}\sum_{e_j\in(\mathbb{E}-E_f)}sim(e_i,e_j)\\1\leq f\leq F
$$

<aside>
💡 이 논문에서 최초에 풀고자 했던 것은 Long Tail로 인해 각 feature embedding이 충분히 학습할 수 있을 만큼 균등하게 샘플링되지 않는 현상이었다. FA/FU Loss를 생성하는 과정에서는 각 feature embedding들이 보다 균등한 횟수로 연산에 참여한다. 따라서 **이것으로 infrequent한 feature embedding이 frequent한 feature embedding과 어느정도의 interaction을 가지게 해 주는 효과가 있을 수 있으며,** 이것은 Long Tail을 극복하는 데에 도움을 줄 수 있다.

</aside>

### 2-4. Final Loss

세 개의 모듈에서 도출된 Loss값을 **하이퍼파라미터** $\alpha$, $\beta$를 이용하여 가중합한다.

$$
\mathcal{L}_{total}=\mathcal{L}_{ctr}+\alpha\cdot\mathcal{L}_{cl}+\beta\cdot(\mathcal{L}_{a}+\mathcal{L}_u)
$$