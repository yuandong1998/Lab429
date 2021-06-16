# 论文阅读《Reinforcement Learning and Additional Rewards for the Traveling Salesman Problem》



## 一、要做什么

​		对于DL+RL求解TSP的问题，通过在训练过程中传递最小生成树信息极高解的质量。还提出了一种能够实时预测TSP实例的最佳长度的深度学习架构。

​		以Deudon等人提出的TSP机器学习算法+Geometric Deep Learning 为基础，目的是了解如何在能够有效解决其他组合优化问题的丰富框架中混合机器学习方法和组合优化概念。



## 二、怎么做

### 2.1 怎么确定reward

​		如下为一个solution的长度。
$$
\mathcal{S}_{\mathrm{L}}(X)=\sum_{i} \sum_{j>i} d_{i j} x_{i j}
$$
​		如下为一个solution占MST的长度的平均值。


$$
\mathcal{R}_{\mathrm{MST}}(X)=\frac{\sum_{\{i, j\} \in \mathcal{M} S \mathcal{T}} x_{i j}}{n}
$$
​		定义最大化总奖励为如下公式，最小化$S_L$，最大化$R_{MST}$：
$$
R_{tot}=R_{MST}-S_L
$$

### 2.2 The model

​		模型采用Encoder-Decoder框架。

#### 2.2.1 Encoder

​		作者认为采用GNN比较好因为有“nodes order invariance”（节点顺序不变性）和“local affinity representation”（局部亲和力表示）的特点。节点顺序不变是输入节点的顺序如何结果一样，这对图来说比较重要。局部相似性是GNN会在所连接节点的空间中创建相似或相近的要素表示。

​		position matrix $P$，经过线性映射到$E$，然后经过transformer堆叠，但在“Attention Layer”[^1]采用“Graph Attention Layer”(GAT)[^2]。Encoder如下图所示。

​		![](https://cdn.mathpix.com/snip/images/QIm3xDH5U1swPt_tsvnYD-q3b1oSGAjkWmOTIIgVTZw.original.fullsize.png)



#### 2.2.2 Decoder

​		$s_{graph}$：the graph information vector，它被定义为加权平均值，其中权重是通过类似于“注意力”的函数（称为“瞥见”）来学习的。

​		$s_{past}$：最后三个动作间隔的编码，表示方向，公式表示为$[e_{m_t},e_{m_{t-1}},e_{m_{t-2}}]$。

​		mask：掩码

​		最后$s_t=[s_{graph},s_{past}]$来表示状态。



### 2.3 训练方法

​		采用带有Baseline的RL方法优化，公式如下。两个$b_{(P)}$函数为critic network function。
$$
\mathbb{E}_{f_{\theta}}\left[\left(\boldsymbol{\Delta} \mathcal{R}_{\mathrm{MST}}-\Delta \mathcal{S}_{\mathrm{L}}\right) \cdot \nabla_{\theta} \log \left(f_{\theta}(\mathbf{X} \mid \mathbf{P})\right)\right.
$$

$$
\begin{aligned}
\boldsymbol{\Delta} \mathcal{R}_{\mathrm{MST}} &=\mathcal{R}_{\mathrm{MST}}(\mathbf{X})-b_{\delta}(\mathbf{P}) \\
\Delta \mathcal{S}_{\mathrm{L}} &=\mathcal{S}_{\mathrm{L}}(\mathbf{X})-b_{\gamma}(\mathbf{P})
\end{aligned}
$$



## 三、结果如何

​		对比EAN，用Z-test来评估p-value。

​		![](https://cdn.mathpix.com/snip/images/BV1ZrT4KZ_Ts9Ee2qU7ChrevxVKOfdWUuLrKlHcJDoQ.original.fullsize.png)





**Reference**

 [^1] P. Velickovi ˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li ´ o, ` and Y. Bengio, “Graph Attention Networks,” International Conference on Learning Representations, 2018. [Online]. Available: https://openreview.net/forum?id=rJXMpikCZ
 [^2] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” CoRR, vol. abs/1409.0473, 2015.  