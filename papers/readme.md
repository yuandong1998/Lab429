# Paper reading

[TOC]

## DRL

|                            Paper                             | Summary                                                      |
| :----------------------------------------------------------: | ------------------------------------------------------------ |
| Machine Learning for Combinatorial Optimization : a Methodological Tour d’ Horizon [[notes]](./paper_notebook/Machine_Learning_for_Combinatorial_Optimization_a_Methodological_Tour_d_Horizon.md) |                                                              |
| Learning TSP Requires Rethinking Generalization [[notes]](./paper_notebook/Learning_TSP_Requires_Rethinking_Generalization.md) | 用机器学习的求解器在小规模上表现良好，但在大规模的问题实例上表现很差。本文主张代替昂贵的大规模训练，从小TSP问题进行高效学习然后通过zero-shot fashion或者fast finetuning迁移到大规模TSP问题，因此确定有前途的归纳偏差，体系结构和学习范式对实现zero-shot泛化有重要作用。<br />本文的目标有两个：（1）在面向规模不变的TSP求解器的端到端学习中，我们统一了几种最新的体系结构和学习范例进入一个实验管道，并为零实例泛化到大型实例提供了第一个原理性研究。 （2）我们开源框架和数据集，以鼓励社区超越评估固定TSP大小的性能，并研究组合问题的转移学习。我们的受控实验表明，在评估泛化时，关于设计选择（例如GNN层，归一化方案，图稀疏度和学习范例）的最佳实践不成立。换句话说，要学习在现实规模上解决TSP，将需要重新考虑神经组合优化的实验和体系结构现状，以明确考虑分布外的泛化。<br />将TSP的解法分为两种，（1）autoregressive approaches，一步一步生成；（2）non-autoregressive 一次生成解决方案。 |
|                                                              |                                                              |



## DRL+local search:

|                            Paper                             | Summary                                                      |
| :----------------------------------------------------------: | :----------------------------------------------------------- |
| LEARN TO DESIGN THE HEURISTICS FOR VEHICLE ROUTING PROBLEM [[notes]](./paper_notebook/LEARN_TO_DESIGN_THE_HEURISTICS_FOR_VEHICLE_ROUTING_PROBLEM.md) | 这篇文章提出了一个学习局部搜索的方法，通过不断迭代解决VRP问题。迭代的过程是将已有解通过destroy算子删除一些节点，然后通过repair算子以最小cost按顺序从后面插入节点。采用了Graph Attention Network集成node和edge嵌入作为encoder，再以GRU作为解码器。通过时间在VRP上有很好的效果，并且可以解决中大规模的问题（400nodes）。 |
| A LEARNING-BASED ITERATIVE METHOD FOR SOLVING VEHICLE ROUTING PROBLEMS [[notes]](./paper_notebook/A_LEARNING_BASED_ITERATIVE_METHOD_FOR_SOLVING_VEHICLE_ROUTING_PROBLEMS.md) | 通过RL+ML+迭代更新 解决CVRP问题，创新点有分开了提升算子和扰动算子，提升算子种类比较多，用规则控制器控制，reward提出了两种方案并对比。 |
| Learning to Perform Local Rewriting for Combinatorial Optimization [[notes]](./paper_notebook/Learning_to Perform_Local_Rewriting_for_Combinatorial_Optimization.md) | In this work, instead of ﬁnding a solution from scratch, we ﬁrst construct a feasible one, then make incremental improvement by iteratively applying local rewriting rules to the existing solution until convergence. |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |
|                                                              |                                                              |



## DRL+CO+生成式：

|                            Paper                             | Summary                                                      |
| :----------------------------------------------------------: | :----------------------------------------------------------- |
| Attention, Learn to Solve Routing Problems! [[notes]](./paper_notebook/Attention_Learn_to_Solve_Routing_Problems.md) | 对于类似TSP的组合优化问题，通过Encoder生成节点嵌入，然后输入decoder生成策略$\pi$，训练方法是 **REINFORCE with a simple baseline based on a deterministic greedy rollout**。可以解决TSP、VRP、OP、PCTSP等问题，在多个问题上使用很灵活。 |
| Coloring Big Graphs With AlphaGo Zero [[notes]](./paper_notebook/Coloring_Big_Graphs_With_AlphaGoZero.md) | 首先用一个深度神经网络将图的状态表示和历史着色记录 $s$ 映射到下一个颜色的概率分布和一个值：$(p,v)=f_{\theta}(s)$，其中 $p$ 表示下一步选择各个可行颜色的概率，$v$ 是一个估计最终结果好于最好启发式方法的标量。然后我们采用MCTS+UCB如图所示搜索标签 $(\pi,z)$，用来对神经网络进行训练。 |
| Reinforcement Learning and Additional Rewards for the Traveling Salesman Problem [[notes]](./paper_notebook/Reinforcement_Learning_and_Additional_Rewards_for_the_Traveling_Salesman_Problem.md) | 对于DL+RL求解TSP的问题，通过在训练过程中传递最小生成树信息极高解的质量。还提出了一种能够实时预测TSP实例的最佳长度的深度学习架构。以Deudon等人提出的TSP机器学习算法+Geometric Deep Learning 为基础，目的是了解如何在能够有效解决其他组合优化问题的丰富框架中混合机器学习方法和组合优化概念。 |





## RL

|                            Paper                             | Summary                                                      |
| :----------------------------------------------------------: | :----------------------------------------------------------- |
| Proximal Policy Optimization Algorithms [[notes]](./paper_notebook/Proximal_Policy_Optimization_Algorithms.md) | 介绍了PPO优化（proximal policy optimization），这是一系列策略优化方法，它们使用多个epochs来随机梯度上升使每个策略更新。这些方法具有信任区方法（trust-region methods）的稳定性和可靠性，但实现起来要简单得多，仅需要几行代码就可以更改为原始策略梯度实现，适用于更常规的设置（例如，策略和价值函数共享参数时）并具有更好的整体效果。 |
| Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor[notes](./paper_notebook/Soft_Actor-Critic.md) |                                                              |
|                                                              |                                                              |



## GNN

|                            Paper                             | Summary                                                      |
| :----------------------------------------------------------: | :----------------------------------------------------------- |
| Graph Attention Networks [[notes]](./paper_notebook/Graph_Attention_Networks.md) | 针对图结构数据，本文提出了一种GAT（graph attention networks）网络。该网络使用masked self-attention层解决了之前基于图卷积（或其近似）的模型所存在的问题。在GAT中，图中的每个节点可以根据邻节点的特征，为其分配不同的权值。GAT的另一个优点在于，无需使用预先构建好的图。因此，GAT可以解决一些基于谱的图神经网络中所具有的问题。实验证明，GAT模型可以有效地适用于（基于图的）归纳学习问题与转导学习问题。 |
|                                                              |                                                              |
|                                                              |                                                              |

