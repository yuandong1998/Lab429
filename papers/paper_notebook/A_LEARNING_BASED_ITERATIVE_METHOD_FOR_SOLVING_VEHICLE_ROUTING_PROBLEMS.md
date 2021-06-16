# 论文阅读《A LEARNING-BASED ITERATIVE METHOD FOR SOLVING VEHICLE ROUTING PROBLEMS》



## 0. Summary

​		通过RL+ML+迭代更新 解决CVRP问题，创新点有分开了提升算子和扰动算子，提升算子种类比较多，用规则控制器控制，reward提出了两种方案并对比。



## 1. Research Objective

​		目的是解决组合优化问题，具体来说是CVRP问题。已经有比较好的传统组合优化求解器，比如LKH3，但是当规模变大的时候需要大量的时间。ML在CO问题上表现不错，但是解的质量没有传统算法高。目的是通过RL+ML方法，从一个随机初始解开始通过迭代不断提升解的质量，解决CVRP问题。



## 2. Problem Statement

​		VRP问题，RL+ML迭代的方法解决。



## 3. Method(s)

<img src="https://s3.ax1x.com/2020/11/20/DQF89P.png" style="zoom:50%;" />



​		上图是问题的整体框架，下面介绍一些详细内容。

* meta-controller：如果L步没有改进则进行Perturbation controller，否则进行 Improvement controller。

* 始终保持解决方案的可行性
* 同时以不同的状态输入特征训练了多个RL策略，然后进行集成，在相同的计算量下比单一策略好
* state有两类，一种是与问题和解相关的，一种是history-related，比如$a_{t-H},1\leq h\leq H$，表示h步前的action，$e_{t-h}$在h步前执行action后改进了solution的话为+1，否则为-1。
* action有很多，分为两类， intra-route operators and inter-route operators。
* 采用reinforce+baseline来更新策略梯度。
* Reward有两种设置方式：（1）如果operator提升了就+1，否则-1。（2）第一次提升的solution的cost设置为baseline，后面的reward是该次迭代的solution的cost与baseline的差值。但是从初始解开始或者经过扰动后的早期提升都比较大和简单，越到后面提升越难切越小，所以给予一样的reward是合适的，也就是方法一。
* 使用$\epsilon$-greedy策略，并训练6种不同的策略，然后进行集成。



<img src="https://cdn.mathpix.com/snip/images/N_uVfCGVTmkj0vmdD7a75U1SeXVYquszILFFirjoVVY.original.fullsize.png" style="zoom:50%;" />



<img src="https://cdn.mathpix.com/snip/images/ZXm_YcVmbueFCOGZyoIhoqqnMemw_Q54h6H91hKEOi8.original.fullsize.png" style="zoom:50%;" />



## 4. Evaluation

$L=6,T=4000,N=20,50,100$，经过L次的更新没有提升则perturb，一共进行T次更新，验证集为2000个随机样例。



**和其他算法对比：**在验证时选择的是6个策略种最小的那个。

![](https://cdn.mathpix.com/snip/images/qkYPZwQRijCJ2ZmdmZIMkaGNdfRwDcKJAWY4vXpabUQ.original.fullsize.png)



**分析集成方法：** 图中上面的蓝线是随机策略，中间是6种其他策略，最下面的是集成策略，可以看出集成策略起到了效果（实验保障了计算量的统一）。

![](https://cdn.mathpix.com/snip/images/d5tTeLgs3wDQAhCiwHMO7DyiPYJmadtXoxjk2H1zadM.original.fullsize.png)



**算子的使用情况**：在采用reward的两种不同方案下，operate使用情况并不相同。

**扰动分析：** 随机排列两个和随机排列所有的扰动方式前者效果更好。



## 5. Conclusion

可以研究在迭代时临时违反约束是否有助于提高解的质量。



## 6. Notes



## Reference



