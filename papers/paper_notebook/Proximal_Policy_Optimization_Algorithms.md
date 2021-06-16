

# 论文阅读《Proximal Policy Optimization Algorithms》



## 0. Summary



## 1. Problem Statement



## 2. Method(s)

### 2.1 Policy Gradient Methods  

策略梯度方法计算策略梯度的估计，然后采用随机梯度下降算法，通常的梯度估计如下：
$$
\hat{g}=\hat{\mathbb{E}}_{t}\left[\nabla_{\theta} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}\right]
$$
$\pi_{\theta}$是一个随机策略（stochastic policy），$\hat{A}_{t}$是在时间步`t`的优势函数，$\hat{\mathbb{E}}_{t}$是在一种在采样（sampling）和优化（optimization）之间交替的算法中计算的有限数量的一批样本的经验平均值。

通过构建一个梯度为策略梯度估计（policy gradient estimator）的目标函数来进行实现。
$$
L^{P G}(\theta)=\hat{\mathbb{E}}_{t}\left[\log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}\right]
$$
对同一个轨迹对$L^{PG}$进行多步的更新虽然很具有吸引力，但这样不合理，从经验上讲，这通常会导致破坏性的大型政策更新。

## 2.2 Trust Region Methods  

TRPO算法的目标函数，约束是为了限制策略更新的幅度。
$$
\begin{array}{ll}
\underset{\theta}{\operatorname{maximize}} & \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right] \\
\text { subject to } & \hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right] \leq \delta
\end{array}
$$
证明TRPO合理的理论实际上建议使用惩罚而不是约束，即解决无约束的优化问题。
$$
\underset{\theta}{\operatorname{maximize}} \hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right]
$$
TRPO算法采用硬约束而不是惩罚项是因为很难选择一个$\beta$对不同的任务都适用。



### 2.3 Clipped Surrogate Objective

TRPO最大化替代目标（surrogate objective），CPI的意思是保守的策略迭代（conservative policy iteration），CPI没有约束，在更新时会造成大的策略更新。
$$
L^{C P I}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\mathrm{old}}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}\right]=\hat{\mathbb{E}}_{t}\left[r_{t}(\theta) \hat{A}_{t}\right]
$$
因此修改目标函数为如下，其中$\epsilon$是超参数设为0.2，
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \hat{A}_{t}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_{t}\right)\right]
$$
通过下图可以看出，该目标函数忽略目标函数提升，包含目标函数下降。（我们的目的就是使得目标函数下降）

<img src="https://s3.ax1x.com/2020/11/20/DQiMWV.png" style="zoom: 67%;" />

下图是沿着更新方向进行插值的几个不同目标的变化，是对练习控制问题进行近端策略优化获得的。



<img src="https://cdn.mathpix.com/snip/images/WTdynT22sz35G7cl6aAuY5XlzSwYSjYy_RSVSspxkd0.original.fullsize.png" style="zoom:67%;" />



### 2.4 Adaptive KL Penalty Coefficient 

* Using several epochs of minibatch SGD, optimize the KL-penalized objective ：

$$
L^{K L P E N}(\theta)=\hat{\mathbb{E}}_{t}\left[\frac{\pi_{\theta}\left(a_{t} \mid s_{t}\right)}{\pi_{\theta_{\text {old }}}\left(a_{t} \mid s_{t}\right)} \hat{A}_{t}-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right]
$$

* Compute  $d=\hat{\mathbb{E}}_{t}\left[\mathrm{KL}\left[\pi_{\theta_{\mathrm{old}}}\left(\cdot \mid s_{t}\right), \pi_{\theta}\left(\cdot \mid s_{t}\right)\right]\right]$
  * If $d<d_{\text {targ }} / 1.5, \beta \leftarrow \beta / 2$
  * If $d>d_{\text {targ }} \times 1.5, \beta \leftarrow \beta \times 2$

Adaptive KL Penalty Coefficient 实验的效果不如Clipped Surrogate Objective。

## 2.5 Algorithm  

如果使用在策略和值函数之间共享参数的神经网络体系结构，则必须使用将策略代理（policy surrogate）和值函数错误项（a value function error term）组合在一起的损失函数，可以通过增加熵值奖金以确保足够的探索来进一步增强该目标[<sup>[1]</sup>](#ref1)[<sup>[2]</sup>](#ref2)。其中$c_1,c_2$是系数，S表示熵，$L_t^{VF}$是$\left(V_{\theta}\left(s_{t}\right)-V_{t}^{\operatorname{targ}}\right)^{2}$。
$$
L_{t}^{C L I P+V F+S}(\theta)=\hat{\mathbb{E}}_{t}\left[L_{t}^{C L I P}(\theta)-c_{1} L_{t}^{V F}(\theta)+c_{2} S\left[\pi_{\theta}\right]\left(s_{t}\right)\right]
$$
采用<sup>[[1]](#ref1)</sup>中的方法，对T个时间步长运行策略（其中T远小于情节长度），并将收集的样本用于更新。这种方法需要一个优势估计量，该估计量不应超过时间步长T，其中t在$[0,T]$中。这个应该注意计算的长度是T-t，而不是T。
$$
\hat{A}_{t}=-V\left(s_{t}\right)+r_{t}+\gamma r_{t+1}+\cdots+\gamma^{T-t+1} r_{T-1}+\gamma^{T-t} V\left(s_{T}\right)
$$
我们可以使用广义优势估计的截断形式，当λ= 1时，可简化为上式。
$$
\begin{array}{l}
\hat{A}_{t}=\delta_{t}+(\gamma \lambda) \delta_{t+1}+\cdots+\cdots+(\gamma \lambda)^{T-t+1} \delta_{T-1} \\
\text { where } \quad \delta_{t}=r_{t}+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)
\end{array}
$$
下面显示了使用固定长度轨迹段的近端策略优化（PPO）算法。

<img src="https://s3.ax1x.com/2020/11/20/DQi7wj.png" style="zoom:67%;" />



## 3. Evaluation

No clipping or penalty、Clipping 、KL penalty(fixed or adaptive)    进行对比，在7个OpenAI Gym的任务上，每个用3个随机数种子，然后分数归一取平均，分数是最后100个episodes的平均总回报。

<img src="https://s3.ax1x.com/2020/11/20/DQijpV.png" style="zoom:67%;" />

和其他算法对比，在几乎所有连续控制环境中，PPO的性能都优于以前的方法。

<img src="https://s3.ax1x.com/2020/11/20/DQF90J.png" style="zoom:67%;" />

## 4. Conclusion

介绍了PPO优化（proximal policy optimization），这是一系列策略优化方法，它们使用多个epochs来随机梯度上升使每个策略更新。这些方法具有信任区方法（trust-region methods）的稳定性和可靠性，但实现起来要简单得多，仅需要几行代码就可以更改为原始策略梯度实现，适用于更常规的设置（例如，策略和价值函数共享参数时）并具有更好的整体效果。



## 5. Notes

[1\]\[2\] 熵值奖金和训练算法。

## Reference

<span id = "ref1">[1] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. “Asynchronous methods for deep reinforcement learning”. In: arXiv preprint arXiv:1602.01783 (2016). </span>

<span id="ref2"> [2] R. J. Williams. “Simple statistical gradient-following algorithms for connectionist reinforcement learning”. In: Machine learning 8.3-4 (1992), pp. 229–256.  </span>

[3] [PPO算法代码demo](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/12_Proximal_Policy_Optimization/simply_PPO.py)

[4] [李宏毅深度强化学习(国语)课程(2018)](https://www.bilibili.com/video/av24724071/?p=4)