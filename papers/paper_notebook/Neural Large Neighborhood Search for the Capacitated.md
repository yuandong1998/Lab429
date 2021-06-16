# Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem



## 0. Summary

写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。



## 1. Problem Statement

问题陈述，需要解决的问题是什么？



## 2. Method(s)

NLNS是对LNS元启发式方法的扩展，充分利用了GPU的并行计算功能，并支持两种搜索模式：批处理搜索（在批处理中可以同时解决一批实例）和单实例搜索（在此情况下可以使用并行解决方案解决单个实例）

LNS是一种元启发式算法，由repair和destory算子组成，最早由【1】提出。

多个repair算子。

初始解采用简单的贪心算法，最近邻。



### 2.1 Destroy operators

- Point-based destroy：removes the customers closest to a randomly selected point from all tours of a solution.
- Tour-based destroy removes all tours closest to a randomly selected point from a solution.



### 2.2  Learning to repair solutions

**IDEA:** 在每个时间步中，模型都将一个不完整的巡回与另一个不完整的巡回或与仓库联系起来。重复该过程，直到解决方案仅包含不超过车辆容量的完整行程。



**输入有哪些**：The model receives a feature vector for each end of an incomplete tour that is not connected to the depot (x1, ..., x5)，Additionally, the model receives a feature vector describing the depot (x0).Furthermore, the model gets one end of a tour as a reference input.. The task of the model is to select where the associated tour end 3 should be connected. 

![](https://i.bmp.ovh/imgs/2021/03/cbf0d37c206b94cb.png)



**不与问题的大小有关，只与destory的程度有关，所以具有很好的扩展性。**



**输入的生成：**

- step1：For each incomplete tour consisting of more than one customer, an input is created for each end that is not the depot
- step2：for each incomplete tour with only one node, a single input is generated
- step3：an input for the depot is created

x为四维：$<x^X,x^Y,x^D,x^S>$ 坐标+sum of the demands fulfilled by the tour。$x^S$为3如果tour包含depot，为2如果不包含，为1如果只有一个点，为-1如果为depot。最后对特征进行归一化。



**迭代过程**

输入表示为$(X_t,f_t)$，分别表示输入特征和reference input。model为$p_\theta(a_t|\pi)$ ，action是$f_t$的连接点。$f_t$在t=0时是随机选取的，之后如果$f_{t-1}$连接后的tour仍然是不完整解，则将$f_t$定为该tour的端点。

通过mask解决不可行的action。



**Model Architecture**

<img src="https://i.bmp.ovh/imgs/2021/03/192825e6af1c4a13.png" style="zoom:67%;" />

采用$Emb_c$ 计算$h_i$，包括two linear transformations with a ReLU activation。$Emb_f$计算$h^f$和$Emb_c$一样。所有$h_i$和$h^f$通过attention layer得出context vector c。计算如下
$$
\begin{aligned}
\bar{a} &=\operatorname{softmax}\left(u_{0}^{H}, \ldots, u_{n}^{H}\right) \\
u_{i}^{H} &=z^{A} \tanh \left(W^{A}\left[h_{i} ; h^{f}\right]\right)\\
c&=\sum_{i=0}^{n} \bar{a}_{i} h_{i}
\end{aligned}
$$
再通过FFN层（a fully connected feed-forward network with two layers both using a ReLU activation）输出q，再和$h_i$进行attention输出再softmax。
$$
\begin{array}{c}
p_{\theta}\left(a_{t} \mid \pi_{t}\right)=\operatorname{softmax}\left(u_{0}, \ldots, u_{n}\right) \\
u_{i}=z^{B} \tanh \left(h_{i}+q\right)
\end{array}
$$

### 2.3 Model Training

目标函数：$J\left(\theta \mid \pi_{0}\right)=\mathbb{E}_{\pi_{T} \sim p_{\theta}\left(. \mid \pi_{0}\right)} L\left(\pi_{T} \mid \pi_{0}\right)$ 其中$L(\pi_T|\pi_0)$表示 the difference between the total tour length of the destroyed solution and the total tour length of the repaired solution. 

对$p_\theta$进行分解，$p_{\theta}\left(\pi_{T} \mid \pi_{0}\right)=\prod_{t=0}^{T-1} p_{\theta}\left(a_{t} \mid \pi_{t}\right)$

采用REINFORCE算法进行训练：$\nabla J\left(\theta \mid \pi_{0}\right)=\mathbb{E}_{\pi_{T} \sim p_{\theta}\left(\cdot \mid \pi_{0}\right)}\left[\left(L\left(\pi_{0}, \pi_{T}\right)-b\left(\pi_{0}\right)\right) \nabla_{\theta} \log p_{\theta}\left(\pi_{T} \mid \pi_{0}\right)\right]$

类似[2] 采用critic网络作为baseline，输入为$\pi_0$生成的$X_0$。The critic processes each input using a position-wise feed-forward network that outputs a continuous value for each input. The sum of all outputs is then the estimate of the repair costs b(π0).



### 2.4 Search Guidance

#### 2.4.1 Single Instance Search

可并行

![](https://i.bmp.ovh/imgs/2021/03/c34e150360aec677.png)

#### 2.4.2 Batch Search

一次搜索batch个实例。



## 3. Evaluation

![](https://i.bmp.ovh/imgs/2021/03/10e76f0415ba7bc5.png)



![](https://i.bmp.ovh/imgs/2021/03/ae9bd6bff03670e9.png)



![](https://i.bmp.ovh/imgs/2021/03/1ee72dd12a36c3be.png)







## 4. Conclusion

作者给了哪些结论，哪些是strong conclusions, 哪些又是weak的conclusions?



## 5. Notes

(optional) 不符合此框架，但需要额外记录的笔记。



## Reference

[1]  P. Shaw. Using constraint programming and local search methods to solve vehicle routing problems. In Fourth International Conference on Principles and Practice of Constraint Programming, pages 417–431. Springer, 1998.

[2]  I. Bello, H. Pham, Q. V. Le, M. Norouzi, and S. Bengio. Neural combinatorial optimization with reinforcement learning. arXiv preprint arXiv:1611.09940, 2016. 

