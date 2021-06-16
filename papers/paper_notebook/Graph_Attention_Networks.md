# 论文阅读《Graph Attention Networks》



## 0. Summary

针对图结构数据，本文提出了一种GAT（graph attention networks）网络。该网络使用masked self-attention层解决了之前基于图卷积（或其近似）的模型所存在的问题。在GAT中，图中的每个节点可以根据邻节点的特征，为其分配不同的权值。GAT的另一个优点在于，无需使用预先构建好的图。因此，GAT可以解决一些基于谱的图神经网络中所具有的问题。实验证明，GAT模型可以有效地适用于（基于图的）归纳学习问题与转导学习问题。



## 1. Research Objective

构建一个处理图结构的神经网络。



## 2. Problem Statement

对GCN引入attention机制。



## 3. Method(s)

​		首先介绍单个graph attention layer，输入是一个节点特征向量集：$\mathbf{h}=\left\{\vec{h}_{1}, \vec{h}_{2}, \ldots, \vec{h}_{N}\right\}, \vec{h}_{i} \in \mathbb{R}^{F}$，其中N是节点个数，F表示输入特征向量维度，输出是一个新的节点特征向量集：$\mathbf{h}^{\prime}=\left\{\vec{h}_{1}^{\prime}, \vec{h}_{2}^{\prime}, \ldots, \vec{h}_{N}^{\prime}\right\}, \vec{h}_{i}^{\prime} \in \mathbb{R}^{F^{\prime}}$,其中$F'$表示新的节点特征向量维度。

​		首先进行self-attention处理，$e_{ij}=a(W\vec h_i,W\vec h_j)$，其中$a$是一个映射到$\mathbb R$的映射，$W\in \mathbb R^{F'*F}$的权重矩阵（共享），最后进行softmax处理。本文采用mask attention将注意力分配到节点i的邻接点上，而不是全部节点，本文中`a`使用单层的前馈神经网络实现，$\vec a^T\in\mathbb R^{2F'}$为前馈神经网络的参数
$$
\alpha_{i j}=\frac{\exp \left(\text { LeakyReLU }\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{j}\right]\right)\right)}{\sum_{k \in \mathcal{N}_{i}} \exp \left(\text { Leaky ReLU }\left(\overrightarrow{\mathbf{a}}^{T}\left[\mathbf{W} \vec{h}_{i} \| \mathbf{W} \vec{h}_{k}\right]\right)\right)}
$$
​		然后本文还采用了多头self-attention，同时计算k个self-attention然后合并（连接或者求和）。
$$
\vec{h}_{i}^{\prime}=\prod_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_{i}} \alpha_{i j}^{k} \mathbf{W}^{k} \vec{h}_{j}\right)
$$


![](https://cdn.mathpix.com/snip/images/7G3OGMOnePZPJmDBUjByJw7FYk3vWzKZWo87qniPZjA.original.fullsize.png)





## 4. Evaluation

* GAT是高效的，无需特征值分解等复杂的矩阵运算，单层GAT时间复杂度为$O(|V|FF'+|E|F')$，与GCN相同。
* 相比于GCN，每个节点的重要性可以是不同的，因此，GAT具有更强的表示能力。
* 对于图中的所有边，attention机制是共享的。因此GAT也是一种局部模型。也就是说，在使用GAT时，我们无需访问整个图，而只需要访问所关注节点的邻节点即可。



## 5. Conclusion

本文提出了一种基于self-attention的图模型。总的来说，GAT的特点主要有以下两点：

- 与GCN类似，GAT同样是一种局部网络。因此，（相比于GNN或GGNN等网络）训练GAT模型无需了解整个图结构，只需知道每个节点的邻节点即可。
- GAT与GCN有着不同的节点更新方式。



## 6. Notes

- **归纳学习（Inductive Learning）：**先从训练样本中学习到一定的模式，然后利用其对测试样本进行预测（即首先从特殊到一般，然后再从一般到特殊），这类模型如常见的贝叶斯模型。
- **转导学习（Transductive Learning）：**先观察特定的训练样本，然后对特定的测试样本做出预测（从特殊到特殊），这类模型如k近邻、SVM等。



## Reference

[1]Veličković P, Cucurull G, Casanova A, et al. Graph Attention Networks. International Conference on Learning Representations (ICLR), 2018.

[2] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. International Conference on Learning Representations (ICLR), 2016.

[3] Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. International Conference on Learning Representations (ICLR), 2017.

[4] Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodol`a, Jan Svoboda, and Michael M Bronstein. Geometric deep learning on graphs and manifolds using mixture model cnns. arXiv preprint arXiv:1611.08402, 2016.

[5] William L Hamilton, Rex Ying, and Jure Leskovec. Inductive representation learning on large graphs. Neural Information Processing Systems (NIPS), 2017.

