# 《Attention, Learn to Solve Routing Problems!》 论文阅读

 [TOC]

## 0. Summary

对于类似TSP的组合优化问题，通过Encoder生成节点嵌入，然后输入decoder生成策略$\pi$，训练方法是 **REINFORCE with a simple baseline based on a deterministic greedy rollout**。可以解决TSP、VRP、OP、PCTSP等问题，在多个问题上使用很灵活。 



## 1. Problem Statement

TSP、VRP、OP、PCTSP。

其中TSP问题每走一步可以看为，从当前点到终点的最短距离，已经走的节点可以忽视。



## 2. Method(s)

本文提出了 Attention Model。模型整体为encoder-decoder框架，其中encoder生成所有节点的嵌入，decoder生成序列$\pi$，一次一个节点，输入为**the encoder embeddings** 、**a problem specific mask**  、**context**  。文中认为已经访问过的其他节点的顺序和坐标无关。解码器的上下文由第一个和最后一个的嵌入组成，用掩码来确定已经访问的节点。



### 2.1 Encoder

<img src="https://cdn.mathpix.com/snip/images/QKROVGS8_Y065BLqtA3oJNBIK8BiOYlZCyh3we5WgSU.original.fullsize.png" style="zoom:50%;" />



​		如上图，encoder采用的是transformer，但没有positional encoding。首先由$x_i$通过$h_i^{(0)}=W^xx_i+b^x$映射到$h_i^{(0)}$，然后通过N个attention layer，最后得到，$h_i^{N}$和$\overline h^{(N)}=\frac{1}{n}\sum h_i^{(N)}$作为decoder的输入。

​		每个attention layer有两个子层，**MHA和FF层**，并且每个sublayer都有**skip-connection**和**batch normalization**，公式表示如下。其中MHA的M=8维度为$\frac{d_h}{M}=16$，FF层有一个维度为512的子层采用Relu激活，如下公式所示。
$$
\begin{aligned}
\hat{\mathbf{h}}_{i} &=\mathrm{BN}^{\ell}\left(\mathbf{h}_{i}^{(\ell-1)}+\mathrm{MHA}_{i}^{\ell}\left(\mathbf{h}_{1}^{(\ell-1)}, \ldots, \mathbf{h}_{n}^{(\ell-1)}\right)\right) \\
\mathbf{h}_{i}^{(\ell)} &=\mathrm{B} \mathrm{N}^{\ell}\left(\hat{\mathbf{h}}_{i}+\mathrm{FF}^{\ell}\left(\hat{\mathbf{h}}_{i}\right)\right)
\end{aligned}
$$



**Attention mechanism:** 

​		其中$W^Q、W^k$为$(d_k * d_h)$，$W^v$为$(d_v * d_h)$，然后的计算如下所示求的$h'$。
$$
\mathbf{q}_{i}=W^{Q} \mathbf{h}_{i}, \quad \mathbf{k}_{i}=W^{K} \mathbf{h}_{i}, \quad \mathbf{v}_{i}=W^{V} \mathbf{h}_{i}
$$

$$
u_{i j}=\left\{\begin{array}{ll}
\frac{\mathbf{q}_{i}^{T} \mathbf{k}_{j}}{\sqrt{d_{k}}} & \text { if } i \text { adjacent to } j \\
-\infty & \text { otherwise }
\end{array}\right.
$$

$$
a_{i j}=\frac{e^{u_{i j}}}{\sum_{j^{\prime}} e^{u_{i j^{\prime}}}}
$$

$$
\mathbf{h}_{i}^{\prime}=\sum_{j} a_{i j} \mathbf{v}_{j}
$$

​		因为是Multi-head attention，M=8，最后计算如下所示。
$$
\mathrm{MHA}_{i}\left(\mathbf{h}_{1}, \ldots, \mathbf{h}_{n}\right)=\sum_{m=1}^{M} W_{m}^{O} \mathbf{h}_{i m}^{\prime}
$$



**Feed-forward sublayer**：

​		FF层有一个维度为512的子层采用Relu激活。
$$
\mathrm{FF}\left(\hat{\mathbf{h}}_{i}\right)=W^{\mathrm{ff}, 1} \cdot \operatorname{ReLu}\left(W^{\mathrm{ff}, 0} \hat{\mathbf{h}}_{i}+\boldsymbol{b}^{\mathrm{ff}, 0}\right)+\boldsymbol{b}^{\mathrm{ff}, 1}
$$

**batch normalization**：

​		我们将BN与可学习仿射变换参数一起使用。
$$
\mathrm{BN}\left(\mathbf{h}_{i}\right)=\boldsymbol{w}^{\mathrm{bn}} \odot \overline{\mathrm{BN}}\left(\mathbf{h}_{i}\right)+\boldsymbol{b}^{\mathrm{bn}}
$$



### 2.2 Decoder

​		![](https://cdn.mathpix.com/snip/images/CvJr8wpyfGC4UScG95GRXhVqkzfBeX9xontOxUDaiXU.original.fullsize.png)

​		

**(1) context embedding**

​		context embedding由第一个和最后一个节点的嵌入与$\overline h(N)$连接，如果为第一步采用可训练的$v^1,v^f$作为可学习的paceholders。


$$
\mathbf{h}_{(c)}^{(N)}=\left\{\begin{array}{ll}
{\left[\overline{\mathbf{h}}^{(N)}, \mathbf{h}_{\pi_{t-1}}^{(N)}, \mathbf{h}_{\pi_{1}}^{(N)}\right]} & t>1 \\
{\left[\overline{\mathbf{h}}^{(N)}, \mathbf{v}^{1}, \mathbf{v}^{\mathrm{f}}\right]} & t=1
\end{array}\right.
$$
​		然后采用（M-head）attention 来计算一个新的节点嵌入$h_c^{(N+1)}$，M=1也就是single attention head 。
$$
\mathbf{q}_{(c)}=W^{Q} \mathbf{h}_{(c)} \quad \mathbf{k}_{i}=W^{K} \mathbf{h}_{i}, \quad \mathbf{v}_{i}=W^{V} \mathbf{h}_{i}
$$


**(2) Calculation of log-probabilities  **

​		M=1，计算如下：
$$
u_{(c) j}=\left\{\begin{array}{ll}
C \cdot \tanh \left(\frac{\mathbf{q}_{(c)}^{T} \mathbf{k}_{j}}{\sqrt{d_{k}}}\right) & \text { if } j \neq \pi_{t^{\prime}} \quad \forall t^{\prime}<t \\
-\infty & \text { otherwise }
\end{array}\right.
$$

$$
p_{i}=p_{\boldsymbol{\theta}}\left(\pi_{t}=i \mid s, \boldsymbol{\pi}_{1: t-1}\right)=\frac{e^{u_{(c) i}}}{\sum_{j} e^{u_{(c) j}}}
$$

### 2.3 训练方法：REINFORCE WITH GREEDY ROLLOUT BASELINE  

​		损失函数为：$\mathcal{L}(\boldsymbol{\theta} \mid s)=\mathbb{E}_{p_{\boldsymbol{\theta}}(\boldsymbol{\pi} \mid s)}[L(\boldsymbol{\pi})]$，其中$L(\boldsymbol{\pi})$表示预期的成本，然后通过带有b(s)的强化学习算法来优化参数，其中$b(s)$是一个solution的成本，这个solution是通过best的模型进行确定性贪心rollout出来的。
$$
\nabla \mathcal{L}(\boldsymbol{\theta} \mid s)=\mathbb{E}_{p_{\boldsymbol{\theta}}(\boldsymbol{\pi} \mid s)}\left[(L(\boldsymbol{\pi})-b(s)) \nabla \log p_{\boldsymbol{\theta}}(\boldsymbol{\pi} \mid s)\right]
$$

​		

​		**动机与分析：** baseline的目标是评估实例s的难度（成本），因此可以看出用模型生成的$L(\pi)$有多好（advantage），所以一个好的baseline可以减小梯度的方差，然后提升训练速度。为了消除baseline的方差，我们通过贪婪地选择具有最大概率的动作来强制结果具有确定性。对于$L(\pi)-b(s)$，如果solution $\pi$是优于baseline则结果为负数，然后导致actions被加强，反之亦然，这种方式像是一种自我提升。

​		**$b(s)$具体策略**：在每一个epoch，我们将当前的训练策略与基线策略进行比较，并且根据10000个验证样例进行t-test（$\alpha$=0.05），如果有改进更新b(s)，更新后会重新采样新的验证样例。

​		**效率提升：**baseline policy在一个epoch是固定的，所以通过large的batch在epoch中进行前馈计算，也可以通过并行计算提高效率。

<img src="https://cdn.mathpix.com/snip/images/DSQy42cmdhS5UHg93rr4B2Hz0KeXf0nuKFiEHzu48tk.original.fullsize.png" style="zoom: 150%;" />

## 3. Evaluation

**参数:**

1. 参数初始化：uniform $-\frac{1}{\sqrt{d}},\frac{1}{\sqrt{d}}$

2. 每个epoch，2500个batch，每个batch 512个实例。
3. 通过对结果质量与计算复杂度之间权衡，encoder的N层数设为3.
4. 在第一个epoch中采用指数基线（$\beta=0.8$）来稳定初始学习。

**test:**

​	每个问题的测试用例有10000个，在测试时有两种解码方法：（1）greddy，（2）sampling 1280个解。



**TSP**

​		TSP的对比结果如下所示

![](https://cdn.mathpix.com/snip/images/QrbPuWdnLCp_o_hF9oqBOtcmb3Q3lG2wb8S0xK5OpNs.original.fullsize.png)



**Attention Model vs. PN**

![](https://cdn.mathpix.com/snip/images/_By02a5gD-bAyOFP5X5jhnCoShgIE5uoeaoPZ1T-7Gs.original.fullsize.png)

## 4. Conclusion

* 使用注意力代替递归LSTM，可并行加快计算速度，且多头注意可以使节点信息通过多个通道进行提取。

**未来研究方向**

​		1、扩展到较大问题实例，已经使用基于图方法取得较好结果，可以对其进行稀疏化提高计算效率。

​		2、许多具有实际重要性的问题都具有通过简单的掩蔽程序无法满足的可行性约束，并且我们认为研究启发式学习和回溯相结合是否能够解决这些问题很有希望。

## 5. Notes



## Reference

[1] Michel Deudon, Pierre Cournut, Alexandre Lacoste, Yossiri Adulyasak, and Louis-Martin Rousseau. Learning heuristics for the TSP by policy gradient. In International Conference on the Integration of Constraint Programming, Artiﬁcial Intelligence, and Operations Research, pp. 170–181. Springer, 2018. 


