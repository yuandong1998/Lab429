# 论文阅读《LEARN TO DESIGN THE HEURISTICS FOR VEHICLE ROUTING PROBLEM》

[TOC]

## 0. Summary

这篇文章提出了一个学习局部搜索的方法，通过不断迭代解决VRP问题。迭代的过程是将已有解通过destroy算子删除一些节点，然后通过repair算子以最小cost按顺序从后面插入节点。采用了Graph Attention Network集成node和edge嵌入作为encoder，再以GRU作为解码器。通过时间在VRP上有很好的效果，并且可以解决中大规模的问题（400nodes）。



## 1. Problem Statement

**VRP问题：** VRP问题定义了一个有向图(directed graph)$G=(N,A)$，其中$i\in N=\{0,1,...,N\}$，0表示仓库（depot），大于0表示客户（customer），$a_{i,j}\in A$ 表示节点`i`至节点`j`的路径的表示。 `i`节点的需求（demand）表示为 $q_i$ ，节点`i`的服务时间窗（time window）从$s_i$开始到$e_i$结束。VRP问题的目标是找到一个或多个哈密顿环（Hamiltonian cycles，每个节点只访问一次的环）满足以下条件：（1）满足所有的需求；（2）K辆车；（3）时间窗和小车载重约束；的情况小cost最小。



## 2. Method(s)



destroy算子:删除solution中的一部分节点，比如10%。

repair算子:删除的节点按顺序以最小cost插入solution。



### 2.1 Embeddings

8维的初始点嵌入和2维的初始边嵌入，然后进行标准化，再通过Graph Attention Network将边和点进一步集成。	

* node embedding：

  1) service starting time si, 

  2) service ending time ei, 

  3) demand qi, 

  4) total demands of the corresponding route, 

  5) sum up of demands of the corresponding route till this node, 

  6) total traveling distance along this route till this node, 

  7) traveling time along this route till this node, and 

  8) the possible forward shift deﬁned in [1]. 

* edge Embedding:

  1) traveling distance along ai,j  

  2) a binary indicator about whether this arc is part of solution or not. 



### 2.2 The Encoder

GAT只是集成节点信息，而VRP的边上有着重要信息，所以引入了EGATE通过Attention同时集成边和点的信息。

首先全连接层扩展原始嵌入：
$$
\begin{aligned}
\tilde{n}_{i} &=W_{n} * n_{i} \\
\tilde{e}_{i, j} &=W_{\text {edge }} * e_{i, j}
\end{aligned}
$$
然后计算pair<i,j>的attention权重：
$$
\begin{aligned}
h_{\text {concat }, i j} &=\operatorname{concat}\left(\tilde{n}_{i}, \tilde{n}_{j}, \tilde{e}_{i, j}\right) \\
w_{i, j} &=\text { LeakyReLU }\left(W_{L} * h_{\text {concat }, i j}\right) \\
\tilde{w}_{i, j} &=\frac{\exp \left(w_{i, j}\right)}{\sum_{j} \exp \left(w_{i, j}\right)}
\end{aligned}
$$
最后输出每个节点更新后的嵌入：
$$
n_{\mathrm{EGATE}, i}=\tilde{n}_{i}+\sum_{j} \tilde{w}_{i, j} \otimes \tilde{n}_{j}
$$
![](https://cdn.mathpix.com/snip/images/rtLaIniINH8B_u9WeltpQKCaIk9_BS0etCahnxKW-fY.original.fullsize.png)



Encoder遵循GAT的屏蔽注意原则，EGATE通过选择性地对其进行掩蔽，允许在信息传播中排除某些边嵌入。同时EGATE也可以通过叠加为多层。最后进入mean-pooling层得到solution的嵌入。



### 2.3 The Decoder

启发式运算符每次都会迭代生成一个有序列表如下的公式一，以下公式一可以转为公式二。所以可以采用RNN作为解码器。以solution embedding作为初始输入，然后解码对`Encoded Node Embedding`采用attention机制，最后输出概率分布。
$$
\mathcal{H}=\pi\left(\left[\eta_{1}, \eta_{2}, \ldots, \eta_{M}\right]\right)
$$

$$
\begin{aligned}
\mathcal{H} &=\pi\left(\eta_{1}\right) \times \pi\left(\eta_{2} \mid\left[\eta_{1}\right]\right) \ldots \times \pi\left(\eta_{M} \mid\left[\eta_{1}, \ldots, \eta_{M-1}\right]\right) \\
&=\pi\left(\eta_{1}\right) \prod_{m=2}^{M} \pi\left(\eta_{m} \mid\left[\eta_{1}, \ldots, \eta_{m-1}\right]\right)
\end{aligned}
$$

![](https://cdn.mathpix.com/snip/images/MP80_0WTZVZuAYpLf_ESZGkJXpt4fNc0w-iL4-nWBhk.original.fullsize.png)

### 2.4 Train the network

VRP成本是**总行驶距离**和**车辆成本**的总和，时间步`t`的`reward`是时间步`t`的`cost`减去时间步`t-1`的`cost`。
$$
\begin{aligned}
\operatorname{Cost}_{V R P}^{(t)} &=\text {Distance}^{(t)}+C \times K^{(t)} \\
r^{(t)} &=\operatorname{cost}_{V R P}^{(t)}-\operatorname{Cost}_{V R P}^{(t-1)}
\end{aligned}
$$


decoder算是一个actor network。value network是一个两层的前馈神经网络第一层是全连接层加ReLU，第二层是一个线性层，输入是`solution embedding`，<u>所以这个输入时decode之前的还是之后的？</u>。



首先：

（1）计算Advantages as the TD error：
$$
\delta_{\mathrm{TD}}^{(t)} \leftarrow r^{(t)}+\gamma \hat{v}\left(E n c^{(t)}, \phi\right)-\hat{v}\left(E n c^{(t-1)}, \phi\right)
$$
（2）训练critic network：
$$
\phi \leftarrow \phi+\alpha_{\phi} \delta_{\mathrm{TD}}^{(t)} \nabla_{\phi} \hat{v}\left(E n c^{(t)}, \phi\right)
$$
（3） 通过 clipped surrogate objective Proximal Policy Optimization (PPO) 方法训练actor，$rt(\theta)$是新策略对旧策略的比值。
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \delta_{\mathrm{TD}}^{(t)}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \delta_{\mathrm{TD}}^{(t)}\right)\right]
$$
​	

最后如果满足以下条件才更新，其中`Rnd`是[0,1]之间的随机数，$T^{(t)}$是模拟退火（SA）$T^{(t)}=\alpha_T T^{(t-1)}$：
$$
\text { Distance }^{(t)}<\text { Distance }^{(t-1)}-T^{(t)} * \log (R n d)
$$
算法流程：

![](https://cdn.mathpix.com/snip/images/Y-wAcb6Mnu4sZvCCcO7bTHxfsxfDiqjcvfCCGgdf3SE.original.fullsize.png)





## 3. Evaluation

将本文模型分别和三种种手工启发式算法（Random、ALNS、SISR）、AM（贪心、sample）模型对比。SISR迭代1M次看作接近最优值的基准。EGATE100-1k中100表示evaluation batch size，1k表示迭代次数。测试的实例超过100个，如果batch_size>1，则每个实例批处理最小成本作为实例的结果。<u>评估里的batch是什么意思？</u>

![](https://cdn.mathpix.com/snip/images/pgQCgHshnD5lz7PlU28WYlnz5ViYr4HXVTqIY9ZRqO8.original.fullsize.png)

​	

对于400nodes的中大规模的问题，图每个节点保留Top10%最近的边和rout的边来简化计算，EGATE的test batch size设置为192。

![](https://cdn.mathpix.com/snip/images/m9kozh_VwqAHuZtBghEUMjq97uah_yUwH_-451Hert4.original.fullsize.png)



## 4. Conclusion

本文方法效果不错。



## 5. Notes



## Reference

**the possible forward shift:**

​	[1]Mwp Martin Savelsbergh. A parallel insertion heuristic for vehicle routing with side constraints. 1990.

**The Graph Attention Network (GAT):**

​	[2] Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017. 

 **clipped surrogate objective Proximal Policy Optimization (PPO) method:**

 	[3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.



