# Learning 2-opt Heuristics for the Traveling Salesman Problem



## 0. Summary



## 1. Problem Statement

问题陈述，需要解决的问题是什么？



## 2. Method(s)

![](https://i.bmp.ovh/imgs/2021/03/7fe8f754ee7428f7.png)

### 2.1 Reinforcement Learning Formulation

**States** ：current solution和best solution$\hat S=(S,S')$

**Actions** ：2-opt

**Rewards** ：当前最优解的差$R_t=L(S')-L(S'_{t+_1})$



### 2.2 Encoder

**Embedding Layer**

二维坐标经过线性层，边经过normalization：$\tilde{e}_{i, j}=\frac{e_{i, j}}{\sqrt{\sum_{j=1}^{n} e_{i, j} \sum_{i=1}^{n} e_{i, j}}}$



**Graph Convolutional Layers**

经过$L$层的GCN。$x_{i}^{\ell+1}=x_{i}^{\ell}+\operatorname{ReLU}\left(\sum_{j \in \mathcal{N}(i)} \tilde{e}_{i, j}\left(W_{g}^{\ell} x_{j}^{\ell}+b_{g}^{\ell}\right)\right)$



**Sequence Embedding Layers**

以GCN的嵌入为输入，进行双向LSTM，



**Dual Encoding** 

对current solution和best solution进行独立编码。



### 2.3 Policy Decoder

参数化策略：$$\pi_{\theta}(A \mid \bar{S})=\prod_{i=1}^{k} p_{\theta}\left(a_{i} \mid a_{<i}, \bar{S}\right)$$

query向量：$q_{i}=\tanh \left(\left(W_{q} q_{i-1}+b_{q}\right)+\left(W_{o} o_{i-1}+b_{o}\right)\right)$，$o_0$是满足uniform分布的参数， $h_{\bar{s}}=W_{s} h_{n}+b_{s} \| W_{s^{\prime}} h_{n}^{\prime}+b_{s^{\prime}}$  ,   $z_{g}=\max \left(z_{1}, \ldots, z_{n}\right)$  ,$q_{0}=h_{\bar{s}}+z_{g}$.

**Pointing Mechanis**

使用一种指向机制来预测给定编码动作（节点）和状态表示形式（查询矢量）的节点输出的分布  $u_{j}^{i}=\left\{\begin{array}{ll}
v^{T} \tanh \left(K o_{j}+Q q_{i}\right), & \text { if } j>a_{i-1} \\
-\infty, & \text { otherwise }
\end{array}\right.$



###  2.4 Value Decoder

计算value值，公式如下：
$$
V_{\phi}(\bar{S})=W_{r} \operatorname{ReLU}\left(W_{z}\left(\frac{1}{n} \sum_{i=1}^{n} z_{i}+h_{v}\right)+b_{z}\right)+b_{r}
$$
$h_{\bar{v}}=W_{s} h_{n}+b_{s} \| W_{s^{\prime}} h_{n}^{\prime}+b_{s^{\prime}}$

### 2.5 Policy Gradient Optimization

policy梯度：
$$
\nabla_{\theta} J(\theta) \approx \frac{1}{B} \frac{1}{T}\left[\sum_{b=1}^{B} \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}\left(A_{t}^{b} \mid \bar{S}_{t}^{b}\right)\left(G_{t}^{b}-V_{\phi}\left(\bar{S}_{t}^{b}\right)\right)\right]
$$
其中$\mathcal{A}_{t}^{b}=G_{t}^{b}-V_{\phi}\left(\bar{S}_{t}^{b}\right)$。

为了避免过早收敛到次优政策增加了熵奖励：$H(\theta)=\frac{1}{B} \sum_{b=1}^{B} \sum_{t=0}^{T-1} H\left(\pi_{\theta}\left(\cdot \mid \bar{S}_{t}^{b}\right)\right)$，其中$H\left(\pi_{\theta}\left(\cdot \mid \bar{S}_{t}^{b}\right)\right)=-\mathbb{E}_{\pi_{\theta}}\left[\log \pi_{\theta}\left(\cdot \mid \bar{S}_{t}^{b}\right)\right.$



value目标：$\left.\mathcal{L}(\phi)=\frac{1}{B} \frac{1}{T}\left[\sum_{b=1}^{B} \sum_{t=0}^{T-1} \| G_{t}^{b}-V_{\phi}\left(\bar{S}_{t}^{b}\right)\right) \|_{2}^{2}\right]$

![](https://i.bmp.ovh/imgs/2021/03/5596fa5a12347037.png)



## 3. Evaluation

对rewards进行裁剪为1，以支持非贪婪行为并稳定学习。

![](https://i.bmp.ovh/imgs/2021/03/4278c5c0386a0cbc.png)



![image-20210301192520896](C:\Users\13775\AppData\Roaming\Typora\typora-user-images\image-20210301192520896.png)



## 4. Conclusion

作者给了哪些结论，哪些是strong conclusions, 哪些又是weak的conclusions?



## 5. Notes

(optional) 不符合此框架，但需要额外记录的笔记。



## Reference

(optional) 列出相关性高的文献，以便之后可以继续track下去。





## 代码

对reward进行了裁剪。

```python
reward = self.current_best_distance - self.tour_distance
reward = round(min(reward/10000, 1.0), 4)
```

