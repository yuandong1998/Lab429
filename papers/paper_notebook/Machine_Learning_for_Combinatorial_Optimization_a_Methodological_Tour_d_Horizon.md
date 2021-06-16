# 《Machine Learning for Combinatorial Optimization : a Methodological Tour d’ Horizon  》阅读笔记



## 2. Preliminaries  

### 2.1 Combinatorial Optimization  

CO问题可以被表述为一个有约束的最小优化问题，约束模型问题的自然的或强加的限制，变量定义要做的决定，而目标函数，通常是成本最小化，定义了每一个可行的变量赋值质量的度量。

* 线性目标规划（LP）：目标和约束是线性的。

* 混合整数线性规划（MILPs）：一些变量被限制为只能为整数

* 可行域（feasible region）：满足约束条件的点集

LP是多项式负责问题，可以通过simplex algorithm  或者interior points methods求解。MILPs是NP难问题，与一些变量要求是整数有关，（B&B算法没看懂）

### 2.2 Machine Learning

####  Supervised learning 

略

#### Unsupervised learning

很少与CO一起使用

#### Reinforcement learning  

在RL中，Agent通过马尔可夫决策过程（MDP）与环境进行交互。在每一个时间步，Agent处于环境给定的状态（State），并根据策略选择一个动作（Action），然后从环境中获得奖励并进入一个新的状态。RL的目标是训练Agent最大化未来回报的期望总和。对于一个策略，给定当前状态（状态动作对）的期望回报被称为值函数（value function）。值函数遵循Bellman方程，因此可以将问题表示为动态规划，并进行近似求解。

![](https://cdn.mathpix.com/snip/images/AoBBx7tfEoDWvE4d0r15d1sxLj0ZiXq2hALQCEnWG1s.original.fullsize.png)

 

#### Deep learning  

RNN、注意力机制、图神经网络

## 3. Recent approaches 

ML结合的两个动机，近似（approximation）与发现新的策略（discovery of new policies）。以及ML和传统CO算法结合的不同方法的例子。

### 3.1  Learning methods  

如同所示，ML在CO中的动机分为两种，一种是近似，在已有专家策略的情况下去逼近专家策略（Demonstration）。第二种是发现新的策略，需要从头开始优化一个策略函数，用RL算法使期望的报酬总和最大化。一个是告诉agent要做什么，一个是激励agent快速积累奖励。

<img src="https://cdn.mathpix.com/snip/images/J3qLaKxYussL41PouqrkwngwREic6hyr5evCZN-p4-s.original.fullsize.png" style="zoom:80%;" />

<img src="https://cdn.mathpix.com/snip/images/SWxmznSL2eCqBArrOB7rvLXfm25q_mdE9MRfczI_uls.original.fullsize.png" style="zoom:80%;" />



#### 3.1.1 Demonstration

#### 3.1.2 Experience  

### 3.2 Algorithmic structure  

#### 3.2.1 End to end learning  

<img src="https://cdn.mathpix.com/snip/images/QqfVB_nKN7E9IYaAMHRNd9aJNe3w0RhpTwgnteNGFbw.original.fullsize.png" style="zoom:80%;" />

Vinyals et al.  等人提出pointer network，然后Bahdanau et al.  等人加入attention mechanism机制，pointer network是通过预先计算的TSP解为目标来训练模型。Bello et al. 使用类似的模型并使用tour length作为奖励信号，通过强化学习来训练，解决了监督学习的的一些限制，比如需要计算最优的或者高质量的TSP解决方案。Kool and Welling  (2018)采用过了GNN、并引入了更多的先验知识。Emami和Ranka (2018)以及Nowak等人(2017)探索了一种不同的方法，通过在神经网络的输出中直接近似双随机矩阵来表征排列。（不太懂）

Larsen(2018)训练一个神经网络来预测一个存在确定的MILP公式的随机负荷规划问题的解，动机是需要在战术层面做出决定。在不完全信息下，用机器学习来处理由于在观测输入中缺少某些状态变量而引起的问题的随机性。文中提出了对解决方案的最高描述是其成本，而最低描述是其所有变量的值的知识。网络还将实例作为一个向量进行处理。（随机负荷规划？什么是tactical level？the stochasticity of the problem ？最低描述是the knowledge of values for all its variables？）

#### 3.2.2 Learning to configure algorithms  

ML可以为CO算法提供额外的信息。

![](https://cdn.mathpix.com/snip/images/FKlOSePV2M0JazXb6zAQGH047p4R-DIc2sjUhTqkIyI.original.fullsize.png)

（没太懂）

#### 3.2.3 Machine learning alongside optimization algorithms  

![](https://cdn.mathpix.com/snip/images/ZDsH-ZoeUfVZQQwAjk_ouwoCp1nRoel-s5YPt645MAA.original.fullsize.png)

## 4 Learning objective  

本节阐述驱动学习过程的目标（learning objective）

## 4.1 Multi-instance formulation  

## 4.2 Surrogate objectives  (代理的目标)















