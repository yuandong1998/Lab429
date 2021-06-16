# 论文阅读 《Learning TSP Requires Rethinking Generalization  》



## 一、论文要做什么

​		用机器学习的求解器在小规模上表现良好，但在大规模的问题实例上表现很差。本文主张代替昂贵的大规模训练，从小TSP问题进行高效学习然后通过zero-shot fashion或者fast finetuning迁移到大规模TSP问题，因此确定有前途的归纳偏差，体系结构和学习范式对实现zero-shot泛化有重要作用。

​		本文的目标有两个：（1）在面向规模不变的TSP求解器的端到端学习中，我们统一了几种最新的体系结构和学习范例进入一个实验管道，并为零实例泛化到大型实例提供了第一个原理性研究。 （2）我们开源框架和数据集，以鼓励社区超越评估固定TSP大小的性能，并研究组合问题的转移学习。

​		我们的受控实验表明，在评估泛化时，关于设计选择（例如GNN层，归一化方案，图稀疏度和学习范例）的最佳实践不成立。换句话说，要学习在现实规模上解决TSP，将需要重新考虑神经组合优化的实验和体系结构现状，以明确考虑分布外的泛化。

​		将TSP的解法分为两种，（1）autoregressive approaches，一步一步生成；（2）non-autoregressive 一次生成解决方案。





## 二、怎么做



### 2.1 Neural Combinatorial Optimization Pipeline  

​		文章设计了一个五个阶段的端到端的pipline，如下图所示。

![](https://cdn.mathpix.com/snip/images/5HGTkHpin4FbcWj5tA-hiIJf5ZXhcMHS2jwkVJQ0IRQ.original.fullsize.png)

**（1）Problem Definition **
$$
L(\pi \mid s)=\left\|x_{\pi_{n}}-x_{\pi_{1}}\right\|_{2}+\sum_{i=1}^{n-1}\left\|x_{\pi_{i}}-x_{\pi_{i+1}}\right\|_{2}
$$


**（2）Graph Embedding  **

​		输入TSP图形中的每个节点计算d维表示，通过节点和边的不断的信息传递。Norm表示归一化层，BatchNorm或者LayerNorm。AGGR表示聚合函数，采样SUM、MEAN、MAX方法。
$$
\begin{aligned}
h_{i}^{\ell+1} &=h_{i}^{\ell}+\operatorname{ReLU}\left(\operatorname{NoRM}\left(U^{\ell} h_{i}^{\ell}+\operatorname{AGGR}_{j \in \mathcal{N}_{i}}\left(\sigma\left(e_{i j}^{\ell}\right) \odot V^{\ell} h_{j}^{\ell}\right)\right)\right) \\
e_{i j}^{\ell+1} &=e_{i j}^{\ell}+\operatorname{ReLU}\left(\operatorname{NoRM}\left(A^{\ell} e_{i j}^{\ell}+B^{\ell} h_{i}^{\ell}+C^{\ell} h_{j}^{\ell}\right)\right)
\end{aligned}
$$


**（3）Solution Decoding  **

Non-autoregressive Decoding (NAR)  ：预测各个边是否属于solution。
$$
\hat{p}_{i j}=W_{2}\left(\operatorname{ReLU}\left(W_{1}\left(\left[h_{G}, h_{i}^{L}, h_{j}^{L}\right]\right)\right)\right), \text { where } h_{G}=\frac{1}{n} \sum_{i=0}^{n} h_{i}^{L}
$$


Autoregressive Decoding (AR)  ：尽管NAR解码器一次生成预测便很快，但它们却忽略了TSP巡视的顺序。
$$
\hat{p}_{i j}=\left\{\begin{array}{ll}
C \cdot \tanh \left(\frac{\left(W_{Q} h_{i}^{C}\right)^{T} \cdot\left(W_{K} h_{j}^{L}\right)}{\sqrt{d}}\right) & \text { if } j \neq \pi_{t^{\prime}} \quad \forall t^{\prime}<t \\
-\infty & \text { otherwise }
\end{array}\right.
$$


Inductive Biases ：NAR方法可以相互独立地进行边沿预测，对非顺序问题（如SAT和MVC）显示出很强的分布外概括性。另一方面，AR解码器内置了顺序/巡回约束，是路由问题的默认选择。尽管两种方法均显示出在不同实验设置下固定和较小TSP尺寸下的最佳性能，但重要的是要公平地比较哪种归纳偏置对于一般化最有用。



**（4）Solution Search  **

​		贪心、beam search、采样。

​		在推理过程中使用大的b进行搜索/采样或局部搜索可能会掩盖架构无法泛化的能力。为了更好地理解泛化，我们专注于使用贪婪搜索和小b = 128的波束搜索/采样。



**（5）Policy Learning  **

​		带有baseline的强化学习。
$$
\nabla \mathcal{L}(\theta \mid s)=\mathbb{E}_{p_{\theta}(\pi \mid s)}\left[(L(\pi)-b(s)) \nabla \log p_{\theta}(\pi \mid s)\right]
$$


## 三、结果如何

![](https://cdn.mathpix.com/snip/images/N7wAy438-Nk1BDdks4XZznMZgJ7ilUikYrMGW5y9hm4.original.fullsize.png)

![](https://cdn.mathpix.com/snip/images/3OMtUf-yw77OSeeb4k4l1bm8e3xKwx4ahSeWB9cq2V0.original.fullsize.png)

![](https://cdn.mathpix.com/snip/images/AVOHLNw9jm8XbsxFTS-fdgeSB_lBmmRmxG764CVwQ5U.original.fullsize.png)

![](https://cdn.mathpix.com/snip/images/qa10XdROc1hMoxPDDLwWIW8P3mIGCqwHZEEr_89piZ0.original.fullsize.png)





什么是zero-shot ？