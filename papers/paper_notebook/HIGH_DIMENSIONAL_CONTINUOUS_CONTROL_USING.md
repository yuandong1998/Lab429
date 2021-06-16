# 论文阅读《HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION》



## 0. Summary

policy gradient methods有两个挑战，一是需要大量的样本，二是输入数据不稳定，难以获得稳定稳定的改进。文章对第一个问题采用<u>价值函数以某种偏见为代价，通过大幅降低政策梯度估算的方差，并采用类似于TD（λ）的优势函数的指数加权估算器。</u>（没懂）。对第二个问题采用trust region optimization。



## 1. Problem Statement

 A key source of difficulty is the long time delay between actions and their positive or negative effect on rewards。一个action的奖励有延迟，这个问题被称为credit assignment problem。

value function是帮助我们能够在延迟奖励到来之前估算某项行动的优劣的方法。

parameterized stochastic policy可以获得无偏估计，不幸的是，梯度估计量的方差随时间范围缩放，因为动作的效果与过去和将来动作的效果混杂在一起。另一种policy gradient algorithms actor-critic method，采用value function而不是经验回报，以引入偏差为代价获得方差较小的估计量。尽管高方差需要使用更多的样本，但偏差会更加有害。

problem：策略梯度算法中的偏差问题。



## 2. Method(s)

基于先前的工作[1] [2]。

### 2.1 PRELIMINARIES

策略梯度方法通过反复估计梯度来最大化预期的总回报，$g:=\nabla_{\theta} \mathbb{E}\left[\sum_{t=0}^{\infty} r_{t}\right]$

有如下多种形式：

![](https://ftp.bmp.ovh/imgs/2021/01/ad178ece47510882.png)

对策略梯度方法的解释：that a step in the policy gradient direction should increase the probability of better-than-average actions and decrease the probability of worse-than average actions.

 advantage function：$A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$，表示whether or not the action is better or worse than the policy’s default behavior。只有当$A^{\pi}(s,a)>0$时才会增加$\pi_{\theta}(a_t|s_t)$。



引入衰减系数，降低与延迟效应相对应的奖励的权重来减少方差，代价是引入偏差。如下是引入偏差后的计算公式。

![](https://ftp.bmp.ovh/imgs/2021/01/9a614859d9c4bb6f.png)

策略梯度的折现近似定义如下：
$$
g^{\gamma}:=\mathbb{E}_{a_{0: \infty}}\left[\sum_{t=0}^{\infty} A^{\pi, \gamma}\left(s_{t}, a_{t}\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right]
$$


$\gamma -just$的定义，找到$A^{\pi,\gamma}$的估计$\hat A_t$，使得用这个估计来计算得到的梯度估计期望不变。如下式子成立。
$$
\mathbb{E}_{a_{0}: \infty}\left[\hat{A}_{t}\left(s_{0: \infty}, a_{0: \infty}\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right]=\mathbb{E}_{a_{0: \infty}}\left[A^{\pi, \gamma}\left(s_{t}, a_{t}\right) \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right)\right]
$$
如下均为$\gamma-just$的。

![](https://ftp.bmp.ovh/imgs/2021/01/ea9d86328345b917.png)

### 2.2 ADVANTAGE FUNCTION ESTIMATION

 the TD residual of V with discount γ：$\delta_{t}^{V}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)$，可以看作是$a_t$的advantage 估计。

只有当$V=V^{\pi,\gamma}$，如下才是$\gamma-just$的，否则会引入偏差。
$$
\begin{equation}
\begin{aligned}
\mathbb{E}_{s_{t+1}}\left[\delta_{t}^{V^{\pi, \gamma}}\right] &=\mathbb{E}_{s_{t+1}}\left[r_{t}+\gamma V^{\pi, \gamma}\left(s_{t+1}\right)-V^{\pi, \gamma}\left(s_{t}\right)\right] \\
&=\mathbb{E}_{s_{t+1}}\left[Q^{\pi, \gamma}\left(s_{t}, a_{t}\right)-V^{\pi, \gamma}\left(s_{t}\right)\right]=A^{\pi, \gamma}\left(s_{t}, a_{t}\right)
\end{aligned}
\end{equation}
$$
如下都可以作为advantage的估计，并且随着k增大偏差在变小。

<img src="https://ftp.bmp.ovh/imgs/2021/01/99f82fed69b3fbff.png" style="zoom:67%;" />

k趋近于无穷时，经验回报减去value function baseline。
$$
\hat{A}_{l}^{(\infty)}=\sum_{l=0}^{\infty} \gamma^{l} \delta_{t+l}^{V}=-V\left(s_{t}\right)+\sum_{l=0}^{\infty} \gamma^{l} r_{t+l}
$$
定义$GAE(\gamma,\lambda)$，is defined as the exponentially-weighted average of these k-step estimators。下面式子与$TD(\lambda)$相似，$TD(\lambda)$是价值函数的估计器，而在这里我们估计的是优势函数。如下有$\lambda=0、1$的特殊情况。为1时具有较高的方差，为0时具有较低的方差。$\lambda$调控偏差与方差，同时$\gamma$也可以平衡偏差与方差，但是用途不同。不管值函数的准确性如何，取γ<1都会在策略梯度估计中引入偏差，仅当值函数不准确时，λ<1才会引入偏差。根据经验，我们发现λ的最佳值远低于γ的最佳值，这可能是因为λ引入的偏差远小于γ的合理准确值函数。
$$
\begin{aligned}
\hat{A}_{t}^{\operatorname{GAE}(\gamma, \lambda)} &:=(1-\lambda)\left(\hat{A}_{t}^{(1)}+\lambda \hat{A}_{t}^{(2)}+\lambda^{2} \hat{A}_{t}^{(3)}+\ldots\right) \\
&=(1-\lambda)\left(\delta_{t}^{V}+\lambda\left(\delta_{t}^{V}+\gamma \delta_{t+1}^{V}\right)+\lambda^{2}\left(\delta_{t}^{V}+\gamma \delta_{t+1}^{V}+\gamma^{2} \delta_{t+2}^{V}\right)+\ldots\right) \\
&=(1-\lambda)\left(\delta_{t}^{V}\left(1+\lambda+\lambda^{2}+\ldots\right)+\gamma \delta_{t+1}^{V}\left(\lambda+\lambda^{2}+\lambda^{3}+\ldots\right)\right.\\
&\left.\quad+\gamma^{2} \delta_{t+2}^{V}\left(\lambda^{2}+\lambda^{3}+\lambda^{4}+\ldots\right)+\ldots\right) \\
&=(1-\lambda)\left(\delta_{t}^{V}\left(\frac{1}{1-\lambda}\right)+\gamma \delta_{t+1}^{V}\left(\frac{\lambda}{1-\lambda}\right)+\gamma^{2} \delta_{t+2}^{V}\left(\frac{\lambda^{2}}{1-\lambda}\right)+\ldots\right) \\
&=\sum_{l=0}^{\infty}(\gamma \lambda)^{l} \delta_{t+l}^{V}
\end{aligned}
$$

$$
\begin{array}{l}
\operatorname{GAE}(\gamma, 0): \hat{A}_{t}:=\delta_{t} \quad=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right) \\
\operatorname{GAE}(\gamma, 1): \hat{A}_{t}:=\sum_{l=0}^{\infty} \gamma^{l} \delta_{t+l}=\sum_{l=0}^{\infty} \gamma^{l} r_{t+l}-V\left(s_{t}\right)
\end{array}
$$

得出$g^{\gamma}$的有偏估计，$\lambda=1$时相等。
$$
g^{\gamma} \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}^{\operatorname{GAE}(\gamma, \lambda)}\right]=\mathbb{E}\left[\sum_{t=0}^{\infty} \nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \sum_{l=0}^{\infty}(\gamma \lambda)^{l} \delta_{t+l}^{V}\right]
$$


### 2.3  INTERPRETATION AS REWARD SHAPING

Reward shaping 指的是对MDP的reward进行如下转换，：  $\Phi: \mathcal{S} \rightarrow \mathbb{R}$ 状态空间映射到标量的函数, 定义转换后的奖励函数 $\tilde r$ 为如下。
$$
\tilde{r}\left(s, a, s^{\prime}\right)=r\left(s, a, s^{\prime}\right)+\gamma \Phi\left(s^{\prime}\right)-\Phi(s)
$$
对任何策略$\pi$的$A^{\pi,\gamma}$保持不变，如下时从$s_t$开始的discounted sum of rewards。
$$
\sum_{l=0}^{\infty} \gamma^{l} \tilde{r}\left(s_{t+l}, a_{t}, s_{t+l+1}\right)=\sum_{l=0}^{\infty} \gamma^{l} r\left(s_{t+l}, a_{t+l}, s_{t+l+1}\right)-\Phi\left(s_{t}\right)
$$
再定义如下Q、A、V的估计。
$$
\begin{array}{l}
\tilde{Q}^{\pi, \gamma}(s, a)=Q^{\pi, \gamma}(s, a)-\Phi(s) \\
\tilde{V}^{\pi, \gamma}(s, a)=V^{\pi, \gamma}(s)-\Phi(s) \\
\tilde{A}^{\pi, \gamma}(s, a)=\left(Q^{\pi, \gamma}(s, a)-\Phi(s)\right)-\left(V^{\pi, \gamma}(s)-\Phi(s)\right)=A^{\pi, \gamma}(s, a) .
\end{array}
$$
使得$\Phi=V$可得：
$$
\sum_{l=0}^{\infty}(\gamma \lambda)^{l} \tilde{r}\left(s_{t+l}, a_{t}, s_{t+l+1}\right)=\sum_{l=0}^{\infty}(\gamma \lambda)^{l} \delta_{t+l}^{V}=\hat{A}_{t}^{\operatorname{GAE}(\gamma, \lambda)}
$$


## 3. Evaluation

略



## 4. Conclusion

略



## 5. Notes

略



## Reference

[1] Kimura, Hajime and Kobayashi, Shigenobu. An analysis of actor/critic algorithms using eligibility traces: Reinforcement learning with imperfect value function. In ICML, pp. 278–286, 1998.

[2] Wawrzynski, Paweł. Real-time reinforcement learning by sequential actor–critics and experience replay. ´ Neural Networks, 22(10):1484–1497, 2009.