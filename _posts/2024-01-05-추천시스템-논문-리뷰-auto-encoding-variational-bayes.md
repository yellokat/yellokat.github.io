---
math: true
layout: post
title: "[추천시스템 논문 리뷰] Auto-Encoding Variational Bayes"
date: 2024-01-05 19:35 +0900
categories:
  - Papers
tags:
  - Recommender Systems
  - Multi-Modal Recommendation
  - Machine Learning
  - Deep Learning
---

### Significance

- 사후확률이 Intractable한 경우에도 작동하는 확률적 변분추론 알고리즘과 학습 알고리즘을 제안한다.

### Scenario

- ${\bf X}=\{\bf{x}^{(i)}\}^N_{i=1}$인 데이터셋 $\bf X$가 존재하며, 이것은 $N$개의 독립동일분포 샘플 $\bf x$로 구성된다.
- 각 데이터포인트 $\bf x$가 어떤 확률과정을 통해 생성된다고 가정하고, 그 확률과정은 연속확률변수 $\bf z$에 영향을 받는다고 하자.
- 그러면 $\bf z$는 사전분포 $p_{\theta^\*}({\bf z})$로부터 생성되고, ${\bf x}^{(i)}$는 $p_{\theta^\*}({\bf x}\|{\bf z})$로부터 생성된다고 말할 수 있다.

### Problem: Intractability

우리는 파라미터 $\theta^\*$에 대한 MAP(최대사후확률) 혹은 ML(최대가능도) 추론을 진행하고 싶다. 문제는 많은 경우 marginal likelihood에 해당하는 $p_\theta({\bf x})=\int p_\theta({\bf z})p_\theta({\bf x}\|{\bf z})d{\bf z}$를 직접 다루기 어렵다는 것이다. $p_\theta(\bf {x})$가 Intractable한 경우, $p_\theta({\bf z}\|{\bf x})=p_\theta({\bf x}\|{\bf z})p_\theta({\bf z})/{p_\theta({\bf x})}$ 역시 Intractable하기 때문에 EM(Expectation Maximization) 알고리즘을 사용할 수 없게 된다. Variational Inference에서 자주 사용되는 Mean-Field 전략 역시 Gradient를 계산할 수 없으므로 사용할 수 없다. **심지어 이 문제는 분포에 대한 통계적 가정을 할 수 없는 현실에서 매우 자주 발생한다.** 예를 들자면 비선형적 활성화 함수를 사용한 인공신경망의 가능도함수는 당연히 Intractable하다.

### Problem: A large dataset

데이터가 너무 큰 경우에는 몬테카를로 방법 등을 사용하려면 비용이 너무나 많이 든다.

### Contribution

- VAE는 이러한 시나리오에서 파라미터 $\bf \theta$에 대한 ML/MAP 추정을 가능케 하고, 파라미터의 근사치를 이용해 **데이터 생성에 사용된 확률과정을 모방하여 새로운 데이터를 생성할 수 있게 된다.**
- 실제 사후확률 $p_\theta({\bf z}\|{\bf x})$를 근사한 $q_\phi({\bf z}\|{\bf x})$를 도입하고, $\phi$와 $\theta$를 동시에 학습하여 문제를 푼다. 이 때 $q_\phi({\bf z}\|{\bf x})$를 encoder라고 부르고, $p_\theta({\bf x}\|{\bf z})$를 decoder라고 부른다.

## The Variational Bound

우리가 다루기 난해했던 Marginal Likelihood $p_\theta({\bf x})$는 각 데이터포인트의 Marginal Likelihood를 모두 합한 것으로 해석할 수 있다. 

$$

\log p_\theta({\bf x}^{(1)}, \dots, {\bf x}^{(N)}) = \sum^N_{i=1}\log p_\theta({\bf x}^{(i)})
$$

그런데 각 데이터포인트의 Marginal Likelihood는 다시 다음과 같이 정리할 수 있다.

$$
\log p_\theta({\bf x}^{(i)}) = \textcolor{skyblue}{D_{KL}(q_\phi({\bf z}|{\bf x}^{(i)})||p_\theta({\bf z}|{\bf x}^{(i)}))} + \textcolor{red}{\mathcal{L}(\theta, \phi; {\bf x}^{(i)})}
$$

수식의 **파란색 항**은 실제 사후확률 $p_\theta({\bf z}\|{\bf x}^{(i)})$와 그 근사치 $q_\phi({\bf z}\|{\bf x}^{(i)})$ 사이의 거리를 나타내는 KL Divergence에 해당하며, 항상 0보다 큰 값임이 보장된다. 따라서 **빨간색 항**은 Point-Wise Marginal Likelihood의 **하한선**이라고 생각할 수 있으며, 다음과 같이 적을 수 있다.

$$
\begin{align*}
\log p_\theta({\bf x}^{(i)}) \geq \textcolor{red}{\mathcal{L}(\theta, \phi; {\bf x}^{(i)})}&= \mathbb{E}_{q_\phi({\bf z}\|{\bf x})}[-\log q_\phi({\bf z}\|{\bf x}) +\log p_\theta({\bf x},{\bf z})]\\&=-D_{KL}(q_\phi({\bf z}\|{\bf x}^{(i)})||p_\theta({\bf z})) + \mathbb{E}_{q_\theta({\bf z}|{\bf x}^{(i)})}[\log p_\theta({\bf x}^{(i)})|{\bf z}]
\end{align*}
$$

이제 이 하한선을 최소화하는 파라미터 $\theta$와 $\phi$를 구하면 된다. 일반적인 역전파의 진행방식에 따라 $\theta$와 $\phi$를 무작위로 초기화하고, $\nabla_\theta \mathcal{L}(\theta, \phi;{\bf x}^{(i)})$와 $\nabla_\phi\mathcal{L}(\theta, \phi;{\bf x}^{(i)})$를 계산해 조금씩 업데이트한다. 여기서 문제가 생기는데,  해석적으로 $\phi$에 대한 미분이 불가능하다. 그렇다고 해서 몬테카를로 방법(즉 데이터셋에서 샘플링하여 해당 수식의 추정치를 계산하는 것)으로 구하면, 그 추정치가 상당히 부정확하다(Exhibits high variance). 여기서 Reparameterization Trick이 등장한다. 

### The Reparameterization Trick

Reparameterization Trick의 목적은 Variational Lower Bound에 대한, $\phi$와 $\theta$에 대해서 모두 미분가능한 추정치(estimate)를 획득하는 것이다.

- 어떤 **적절한🤔** 분포를 따르는 노이즈 $\epsilon \sim p(\epsilon)$이 존재한다고 가정하자.
- 어떤 **적절한🤔** 함수 $g_\phi(\bf \epsilon, x)$가 존재한다고 가정하자.
- 이제 $\tilde{\bf z} \sim q_\phi(\bf z|x)$라는 사후분포를 $\tilde{\bf z} = g_\phi(\bf \epsilon, x)$라고 표현할 수 있다.
요점은 $\sim$기호가 $=$기호로 바뀌었다는 것이다. 이는 Stochastic했던 $\tilde{\bf z}$를 Deterministic한 값으로 적을 수 있음을 뜻한다. 하지만 $\tilde{\bf z}$에 내재된 Stochasticity가 완전히 사라진 것은 아니다. 그 무작위성에 기여하는 부분이 단순히 $\epsilon$으로 옮겨갔을 뿐이다.
- 마지막으로 $\bf z$에 대한 어떤 함수 $f(\bf z)$가 있다고 하면, $\mathbb{E}\_{q_\phi({\bf z\|x^{(i)}})}[f(\bf z)]$의 몬테카를로 추정치를 다음과 같이 구할 수 있다.
    - ${\bf z}=g_\phi(\bf \epsilon, x)$로 설정하였다.
    - $q_\phi({\bf z\|x})\prod_i dz_i = p(\epsilon)\prod_id\epsilon_i$인 것이 자명하다.
    - 따라서 $\int q_\phi({\bf z\|x})f({\bf z})d {\bf z} = \int p(\epsilon)f({\bf z})d\epsilon = \int p(\epsilon)f(g_\phi({\bf \epsilon, x}))d\epsilon$이다.
    - 이것의 몬테카를로 추정치는 $\int q_\phi({\bf z\|x})f({\bf z})d {\bf z} \simeq \frac{1}{L}\sum^L_{l=1}f(g_\phi({\bf x}, \epsilon^{(l)}))$이다.
    - 이 추정치는 $\theta$와 $\phi$에 대해서 모두 미분가능하다!

### 적절한 선택

**적절한🤔** 함수 $g_\phi(\bf \epsilon, x)$와 노이즈 $\epsilon \sim p(\epsilon)$를 선택하는 방법은 다양하다.

1. **사후확률분포를 tractable한 분포로 설정했을 경우.**
- 함수 $g_\phi(\bf \epsilon, x)$를 $q_\phi({\bf z\|x})$의 inverse CDF로 설정하고
- 노이즈 $\epsilon \sim p(\epsilon)$를 $\mathcal{U}(\bf 0, I)$로 설정할 수 있다.
2. **사후확률분포를 Gaussian(혹은 모든 location-scale 확률분포)으로 설정했을 경우.**
- 함수 $g(\epsilon, \bf x)$를  $z=\mu+\sigma\epsilon$로 설정할 수 있다.
- 노이즈 분포를 $\epsilon \sim \mathcal{N}(0, 1)$로 설정하고
3. **사후확률분포를 다른 확률분포로 분해할 수 있는 확률분포로 설정했을 경우.**
- Log-Normal 분포의 경우, 정규분포의 exponentiation으로 표현된다.
- 감마분포의 경우, 지수족 확률분포들의 합으로 표현된다.
- 디리클레 분포의 경우, 감마분포의 가중합으로 표현된다.
- 베타 분포, 카이제곱분포, F분포 등…
4. **모든 접근이 실패한다면…**
- inverse CDF를 근사해서 진행할 수 있다.
- 이 경우의 시간복잡도는 PDF를 사용할 때와 크게 다르지 않을 것이다.