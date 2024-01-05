---
math: true
layout: post
title: "[추천시스템 논문 리뷰] Factorization Machines"
date: 2024-01-04 13:45 +0900
categories:
  - Papers
tags:
  - Recommender Systems
  - Machine Learning
  - Optimization
  - Linear Algebra
---
## Prerequisites
- Support Vector Machines
- Matrix Factorization for Recommender Systems

## Contribution

**SVM과 같이 General Predictor로 사용될 수 있는 추천 모델을 발표한다.**
- FM은 (MF계열 알고리즘과 다르게) General Predictor 이다.
- FM은 (SVM과 다르게) 모든 feature 사이의 interaction을 모델링한다.
- FM은 (SVM과 다르게) Sparse한 데이터에서도 잘 작동한다.
- FM은 (SVM과 다르게) Space/Time Complexity가 선형적이다.
- FM은 현존하는 많은 Collaborative Filtering 알고리즘의 일반화이다.
    - Biased MF, SVD++, PITF, FPMC 등

{: .highlight }
> \\
> 💡 **그래도 SVM은 근본이다**\\
> SVM은 여전히 인기가 많은 알고리즘이지만, 추천시스템의 도메인에서만큼은 Matrix Factorization 계열 알고리즘에 비해 약세이다. 이것은 추천 도메인의 Sparse한 데이터가 고차원의 Kernel Space에서의 학습을 매우 어렵게 하기 때문이다.

## Proposed Model

### Model Parameters
    
Factorization Machine은 다음과 같은 파라미터들로 구성된다.

$$
\displaylines{
    w_0\in \mathbb{R}\\
    {\bf w} \in \mathbb{R}^{n}\\
    {\bf V} \in \mathbb{R}^{n\times k}
}
$$

여기서 $n$은 입력 Feature Vector의 길이이고, $k$는 Latent Vector의 길이이다. $k\in\mathbb{N}^+_0$는 하이퍼파라미터이다.
    
### Model Equation

$$
\hat{y}({\bf x}) := w_0 + \color{red}{\sum_{i=1}^n w_i x_i}\color{black} + \color{blue}\sum_{i=1}^n \sum_{j=i+1}^n\langle{\bf v}_i,{\bf v}_j\rangle x_i x_j\color{black}
$$

Factorization Machine은 입력 벡터 $\bf x$를 받았을 때, 위와 같은 수식을 거쳐 Prediction $\hat{y}(\bf x)$를 생성한다. **이는 Linear Time 내에 계산이 가능하다.**

- **검은색** 수식은 스칼라항으로, Global Bias를 모델링한다.
- **붉은색** 수식은 입력벡터 $\bf x$와 파라미터 $\bf w$사이의 dot product로, 입력벡터의 각 원소가 어느정도의 중요도를 가지는지를 모델링한다.
- **파란색** 수식은 입력벡터 $\bf x$의 각 원소 $\{x_0, \dots, x_n\}$들이 서로와 interaction하는 정도를 모델링한다.
    
    {: .highlight }
    > \\
    > 💡 **이 부분이 Factorization 요소가 등장하는 부분 중 하나이다.**\\
    > $x_i$와 $x_j$사이의 관계를 나타내는 스칼라 $w_{i,j}$를 $w_{i,j}\in\mathbb{R}$로 모델링할 수도 있지만, 이것을 굳이 두 벡터의 dot product로 표현한 것이다.
    
        
        
### Making Predictions & Learning Factorization Machines
Factorization Machine은 SVM과 같이 General Predictor로써, 여러 종류의 작업을 수행할 수 있다. 각 작업에 따른 목적함수와 손실함수는 다음과 같다.

- **Regression**
$\hat{y}({\bf x})$를 predictor로 사용하고, Squared Error를 최소화.
- **Binary Classification**
$\hat{y}({\bf x})$의 부호(+/-)를 predictor로 사용하고, Hinge/Logit Loss를 최소화.
- **Ranking**
두 개의 입력을 통과시켜 두 개의 $\hat{y}({\bf x})$를 얻고, Pairwise Loss를 최소화.

목적함수와 손실함수를 결정했다면 Gradient Descent를 사용해 학습한다.