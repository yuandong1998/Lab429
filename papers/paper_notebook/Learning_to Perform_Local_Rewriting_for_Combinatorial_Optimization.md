# 论文阅读《Learning to Perform Local Rewriting for Combinatorial Optimization》

[TOC]

## 一、做什么



> In this work, instead of ﬁnding a solution from scratch, we ﬁrst construct a feasible one, then make incremental improvement by iteratively applying local rewriting rules to the existing solution until convergence. 



## 二、怎么做

### 2.1 总体概述

​		整体框架如下，有两步选择，先选择region，再选择rule：

![](https://cdn.mathpix.com/snip/images/vlRQmdE35kTG28uPPlayAgjDO8KWPsbq3CUgVVV42FE.original.fullsize.png)



​		S是问题域的所有可行解，$c:S->\R$为cost function，目标是找到$\arg min_{s\in S}c(s)$。


$$
r(s_t,(w_t,u_t))=c(s_t)-c(s_{t+1})
$$

​		有两个策略，一个是region-picking policy$\pi_\omega$，一个是rule-picking policy $\pi_\mu$。对于$\pi_\omega(w_t|s_t;\theta)$是一个$Q(s_t,w_t;\theta)$的softmax。
$$
\pi_{\omega}\left(\omega_{t} \mid s_{t} ; \theta\right)=\frac{\exp \left(Q\left(s_{t}, \omega_{t} ; \theta\right)\right)}{\sum_{\omega_{t}} \exp \left(Q\left(s_{t}, \omega_{t} ; \theta\right)\right)}
$$
​		其中$Q(s_t,w_t;\theta)$通过如下公式来进行训练，其中T是episod的长度，$\gamma$是decay factor，
$$
L_{\omega}(\theta)=\frac{1}{T} \sum_{t=0}^{T-1}\left(\sum_{t^{\prime}=t}^{T-1} \gamma^{t^{\prime}-t} r\left(s_{t}^{\prime},\left(\omega_{t}^{\prime}, u_{t}^{\prime}\right)\right)-Q\left(s_{t}, \omega_{t} ; \theta\right)\right)^{2}
$$
​		对于$\pi_\mu(\mu|s_t[w_t];)$采用Advantage Actor-Critic训练，以$Q(s_t,w_t;\theta)$为critic，其中advantage function为如下公式1，损失函数如下公式2。
$$
\Delta\left(s_{t},\left(\omega_{t}, u_{t}\right)\right) \equiv \sum_{t^{\prime}=t}^{T-1} \gamma^{t^{\prime}-t} r\left(s_{t}^{\prime},\left(\omega_{t}^{\prime}, u_{t}^{\prime}\right)\right)-Q\left(s_{t}, \omega_{t} ; \theta\right)
$$

$$
L_{u}(\phi)=-\sum_{t=0}^{T-1} \Delta\left(s_{t},\left(\omega_{t}, u_{t}\right)\right) \log \pi_{u}\left(u_{t} \mid s_{t}\left[\omega_{t}\right] ; \phi\right)
$$

​		最后overall loss function为如下公式，其中$\alpha$为超参数。
$$
L(\theta, \phi)=L_{u}(\phi)+\alpha L_{\omega}(\theta)
$$


### 2.2 细节

​		Vehicle Routing Problem的节点嵌入有7维，以一个bi-lstm加上$L_p$层的全连接层来计算Score，rewrite是选择一个点移到另一个节点的后面，先选择一个点$v_j$，然后通过attention 的方法找出另一个节点$v_j'$，训练的过程如下所示：

![](https://cdn.mathpix.com/snip/images/gB_mGbUxPXqY_lZy_I9sE8IPYoH9PSQQS_GsQan8g7I.original.fullsize.png)

参数：

​		$T_{iter}=200$，$p_c$初始化为0.5，每1000步衰减0.8直到值为0.01。$\alpha=10.0,\gamma=0.9$初始学习率为$1e-4$每1000步衰减0.9，batch_size=128。



## 三、结果

![](https://cdn.mathpix.com/snip/images/K07tUFwOX3C0VdYtBJa8cWAnlZYeQuZDlJWnkLWHv3w.original.fullsize.png)

![](https://cdn.mathpix.com/snip/images/4xh0uuyUbuCPCCNQxCF0l1pfgZIt0gu0lVhgv5I1DDA.original.fullsize.png)





soft-Q learning