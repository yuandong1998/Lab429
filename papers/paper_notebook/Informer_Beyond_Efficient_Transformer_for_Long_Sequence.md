# 《Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting》论文笔记  

## 0. Summary

写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。



## 1. Problem Statement

Vanilla Transformer 在LSTF上的三个限制：

* The quadratic computation of self-attention.   self-attention的原子操作canonical dot-product  的时间和空间复杂度为每层$O(L^2)$
* The memory bottleneck in stacking layers for long inputs.   J layer消耗$O(J*L^2)$的内存，扩展性不好。
* The speed plunge in predicting long outputs.   Transformer还是step-by-step inference  和RNN模型一样。



## 2. Method(s)

* `ProbSparse` Self-attention mechanism $O(L\lg L)$的时间和空间复杂度。
* Self-attention Distilling operation ，降低空间复杂度为$O((2-\epsilon)L\lg L)$
* Generative Style Decoder to acquire long sequence output，  只需要一个forward step，同时避免累积误差扩散。



**Efficient Self-attention Mechanism**  

canonical self-attention 由(q,k,v)组成，$A(Q,K,V)=Softmax(\frac{QK^T}{\sqrt k})V$，第`i`个query的attention可以被写为概率形式的内核平滑器：$\mathcal{A}\left(\mathbf{q}_{i}, \mathbf{K}, \mathbf{V}\right)=\sum_{j} \frac{k\left(\mathbf{q}_{i}, \mathbf{k}_{j}\right)}{\sum_{l} k\left(\mathbf{q}_{i}, \mathbf{k}_{l}\right)} \mathbf{v}_{j}=\mathbb{E}_{p\left(\mathbf{k}_{j} \mid \mathbf{q}_{i}\right)}\left[\mathbf{v}_{j}\right]$

内核为指数核$exp(\frac{q_ik_j^T}{\sqrt d})$，self-attention需要$O(L_QL_K)$的内核和二次时间点积计算。

对self-attention进行**定性**评估，具有稀疏性，得分具有长尾分布，少数有贡献，其他可忽略，然后我们需要来区分出有贡献和可忽略的部分。



**Query Sparsity Measurement**   

第i个query的稀疏性评估为：$M\left(\mathbf{q}_{i}, \mathbf{K}\right)=\ln \sum_{j=1}^{L_{K}} e^{\frac{\mathbf{q}_{i} \mathbf{k}_{j}^{\top}}{\sqrt{d}}}-\frac{1}{L_{K}} \sum_{j=1}^{L_{K}} \frac{\mathbf{q}_{i} \mathbf{k}_{j}^{\top}}{\sqrt{d}}$



**ProbSparse Self-attention**  

$\mathcal{A}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{Softmax}\left(\frac{\overline{\mathbf{Q}} \mathbf{K}^{\top}}{\sqrt{d}}\right) \mathbf{V}$  $\overline{\mathbf{Q}}$是稀疏矩阵，仅包含稀疏评估下$M(q,M)$下TOP-u的query，由采样factor c调整，$u=c*\ln L_Q$,这样对每个query-key只需要计算$O(\ln L_Q)$的内积，内存的使用包含$O(L_K\ln L_Q)$，但是计算$M(q_i,K)$的时候为$O(L_QL_K)$，本文提出了query sparsity评估的近似：$\bar{M}\left(\mathbf{q}_{i}, \mathbf{K}\right)=\max _{j}\left\{\frac{\mathbf{q}_{i} \mathbf{k}_{j}^{\top}}{\sqrt{d}}\right\}-\frac{1}{L_{K}} \sum_{j=1}^{L_{K}} \frac{\mathbf{q}_{i} \mathbf{k}_{j}^{\top}}{\sqrt{d}}$ 可以将时间空间复杂度控制到$O(L\ln L)$。



**Encoder: Allowing for processing longer sequential inputs under the memory usage limitation**  

![](https://i.bmp.ovh/imgs/2021/02/1ac0fdff799cc455.png)



**Decoder: Generating long sequential outputs through one forward procedure**  



## 3. Evaluation

作者如何评估自己的方法，实验的setup是什么样的，有没有问题或者可以借鉴的地方。



## 4. Conclusion

作者给了哪些结论，哪些是strong conclusions, 哪些又是weak的conclusions?



## 5. Notes

(optional) 不符合此框架，但需要额外记录的笔记。



## Reference

(optional) 列出相关性高的文献，以便之后可以继续track下去。

文章标题



## 0. Summary

写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文。



## 1. Problem Statement

问题陈述，需要解决的问题是什么？



## 2. Method(s)

作者解决问题的方法/算法是什么？是否基于前人的方法？



## 3. Evaluation

作者如何评估自己的方法，实验的setup是什么样的，有没有问题或者可以借鉴的地方。



## 4. Conclusion

作者给了哪些结论，哪些是strong conclusions, 哪些又是weak的conclusions?



## 5. Notes

(optional) 不符合此框架，但需要额外记录的笔记。



## Reference

(optional) 列出相关性高的文献，以便之后可以继续track下去。