# 论文阅读 《Coloring Big Graphs With AlphaGoZero 》

[TOC]

## 0. Summary

​		首先用一个深度神经网络将图的状态表示和历史着色记录 $s$ 映射到下一个颜色的概率分布和一个值：$(p,v)=f_{\theta}(s)$，其中 $p$ 表示下一步选择各个可行颜色的概率，$v$ 是一个估计最终结果好于最好启发式方法的标量。然后我们采用MCTS+UCB如图所示搜索标签 $(\pi,z)$，用来对神经网络进行训练。

![](https://cdn.mathpix.com/snip/images/moDCkM0S4eznXiEgyEnutZX5zZVF-4RCkmjeXYdjxbM.original.fullsize.png)

## 1. Research Objective

​		在没有先验知识的情况下，利用**AlphaGoZero+graph embedding**学习到一个算法可以在$O(V)$的时间和空间内解决图着色问题。



## 2. Problem Statement

​		图着色问题：给图$G=<V,E>$的每个顶点分配一个颜色，相邻顶点的颜色必须不同，求最少的颜色数。这是一个NP-hard问题，我们在本文采用矩阵 $C$ 表示图 $G$ 的着色情况，其中$C_{i,j}$表示唯一的颜色 $j$ 赋予了$G$中的顶点 $i$ 。



### 2.3 和其他问题进行比较

​		比较MDP的states数和每次game的move数，也就是做决策的次数，可以看出图着色问题的难度远大于其他问题。而且与GO相比，图着色问题具有表示不同图结构和大小的挑战。

![](https://cdn.mathpix.com/snip/images/MiTLIrd2WAJ7yyL3Q-vadp74B7VPEIO6RnQzRxSGIb4.original.fullsize.png)



### 2.1 将图着色问题看为MDP  

​		为了采用强化学习算法，我们需要将图着色问题转为MDP的问题。

* $C^{(t)}$ ：表示在 t 步的state $s^{(t)}$
* $A_i$ ：表示在 t 步可以对下一个节点 i 着色的可行的颜色集合
* $R_a$： 已经使用的颜色的数量的负数
* $\pi_*(s)$：想要学习的最优策略
* $V(s_t)$：遵循$\pi_*$在$s_t$时的期望回报



### 2.2 图着色的self-play 

​		受到self-play的启发定义reward，首先最好的算法来给图着色，然后一个新的算法如果用的颜色更少则赢，否则平局或者输。



## 3. Method(s)

​		首先用一个深度神经网络将图的状态表示和历史着色记录 $s$ 映射到下一个颜色的概率分布和一个值：$(p,v)=f_{\theta}(s)$，其中 $p$ 表示下一步选择各个可行颜色的概率，$v$ 是一个估计最终结果好于最好启发式方法的标量。然后我们采用MCTS+UCB如图所示搜索标签 $(\pi,z)$，用来对神经网络进行训练。

![](https://cdn.mathpix.com/snip/images/4R5X3b6rEOcs5dJZ3wsTQcGO3e4Br-WZzhJNaRbtyGA.original.fullsize.png)

​		

### 3.1 Monte Carlo Tree Search  

![](https://cdn.mathpix.com/snip/images/M4BnFew3vP5YXGl7kpMzcgXK_nDW13PyF3EYh_3T5F0.original.fullsize.png)

​		首先搜索树存储了先验概率 $P(s,a)$ ，访问次数 $N(s,a)$ ，状态行为价值 $Q(s,a)$ 

* **selection**：首先从跟节点开始按照最大化 $Q(s,a)+U(s,a)$ 开始
* **expansion**：直到到达如图所示节点需要扩展
* **simulation**：模拟评估，$(P(s_l,*),V(s_l))=f_{\theta}(s_l)$ 
* **backpropagation**：更新访问的 $N(s,a)$，以及$Q(s,a)=\frac{Q(s,a)*N(s,a)+V(s_l)}{N(s,a)+1}$ 

　<font color=red>在什么情况下会进行扩展，为什么前两层不扩展，后面扩展</font>



### 3.2 Upper Confidence Bound  

​		如下计算$U(s,a)$，其中 $M$ 是当前state的可行action，$c$ 为超参数。MCTS+UCB算法，是比较低的visit count和高的先验概率，更多的是在exploration，而最后有着比较大的 $Q(s,a)$ ，更多的是在exploitation。
$$
U(s, a)=c * P(s, a) * \frac{\sqrt{\sum_{i=b}^{M} N(s, b)}}{1+N(s, a)}
$$


### 3.3 Self-Play  

​		为了生成标签$(\pi,z)$，首先用当前最好的模型生成baseline socre：$\chi (G)$，然后MCTS+UCB用来对同一个图着色，并和baseline比较。在前几次的决策时采用sampling的方法来鼓励exploration，然后再用最大化的方法。我们采用alpha-beta-pruing来做裁剪，self-play的每一次move结果存储为$(G,C,\pi,z)$。神经网络的训练数据是从最近moves 的relay buffer中采样的。

​		由于规模较大每次move都运行完全的MCTS是不可行的，所以提出了一下两个方法：

* **Limited-Run-Ahead**：限制了MCTS的moves次数，为了评估$z$，baseline也限制move次数然后对比。
* **Move-Sampling**：为了生成各种标签，对一个图的所有的moves进行采样然后用来训练，选择动作后，我们将MCTS连续运行几次以避免重置搜索。

### 3.4 FastColorNet  

​		为了提高速度需要（1） 可扩展的消息传递算法；（2）动态大小的softmax。

​		如下图所示的模型预测了 $P$ 和 $V$ ,其中 $P$ 是当前节点的染色概率分布，是大小可变的，而$V$是预期的结果。通过**embedding的随机选择** 和 **extension of truncated back propagation through time** 保持对每个节点的计算需求不变。

![](https://cdn.mathpix.com/snip/images/0O_nPuG7BEePA6bX5B9vL5smZHEAiHNFkvsbBgPILSQ.original.fullsize.png)



#### 3.4.1 Graph Embeddings  

​		采用**传播方向采样**的和**截断的反向传播**（truncated back propagation）的方法，我们只沿一个随机游走应用反向传播，该游走结束于每个引用的嵌入顶点（referenced vertex embedding）。

![](https://cdn.mathpix.com/snip/images/a0QzwB-yhkwAeZSRi0lrkB6ziMxdG2oB706GrpX571o.original.fullsize.png)

　<font color=red>引用的嵌入顶点是什么</font>

#### 3.4.2 Inputs and Outputs 

​		模型输出下一次move的$(p,v)$，标签$(\pi,z)$，损失函数的计算为：
$$
L(\boldsymbol{\pi}, \boldsymbol{p}, z, v)=\boldsymbol{\pi}^{T} \log (\boldsymbol{p})+z^{T} \log (v)
$$
​		**Graph context：** 包含以下单热编码值(G中的顶点数、分配颜色的总数、已经着色的顶点数)与当前顶点的多热编码的有效颜色集连接在一起。

​		**Problem context：**  包含刚刚着色的顶点的嵌入，并且按照着色顺序排列。Problem context在图的开始和结束处填充为零。Problem context中包含的顶点数量是一个超参数，在实验中我们通常将其设置为8。

​		**Possible colors context：** 包含固定大小的顶点集的嵌入，这些顶点集已经用每种可能的颜色着色。如果没有足够的顶点被分配来为一种可能的颜色填充一个完整的集合，那么可能的颜色上下文被填充为零。设置大小是一个超参数，我们在实验中通常设置为4。



![](https://cdn.mathpix.com/snip/images/cWgaFM8FgUe8PUGg_2LWYkhUUc93jbSQ-zbMlYqgKWA.original.fullsize.png)

#### 3.4.3 V-Network  

​		如图a所示。

#### 3.4.4 P-Network  

​		我们从指针网络中获得灵感，并用之前被分配相同颜色的顶点的嵌入来表示颜色。在这个类比中，P-Network选择一个指针指向一个以前有颜色的节点，而不是直接预测可能的颜色。但是，我们的方法与指针网络不同，因为它在同一时间考虑一组具有相同颜色的指针，而不是单个指针。它的不同还在于它考虑的是一组固定的可能的颜色，而不是以前遇到的所有顶点。这些变化对于利用具有相同颜色的节点之间的局部性、提高精确度以及将非常大的图的计算要求限定为图的顺序的线性时间而不是指针网络的二次时间都是很重要的。为了支持可能的颜色的动态数量，首先对每种颜色进行独立处理，生成一个未标准化的分数。然后通过一个序列到序列的模型对该分数进行后期处理，该模型包含了对其他可能颜色的依赖关系。最后的分数由softmax操作规范化。



### 3.5 High Performance Training System  

![](https://cdn.mathpix.com/snip/images/9Mmu_tQp2mZOy0HSBbXKPuWnZtKYjz9ZC8eOy70h_14.original.fullsize.png)

## 4. Evaluation



## 5. Conclusion

​		深度强化学习可以应用于大规模问题，如图着色。

## 6. Notes



## Reference

[1] Alphago 论文

[2] alphago zero 论文

[3] 可变大小的softmax

