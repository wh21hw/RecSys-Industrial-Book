# GLM24级 四次考试题目提取

---

## 第一次考试（glinear20250415）

判断以下各题的陈述是否正确。分数计算公式为 $m-n$，其中 $m$ 为判断结论"正确"的题目总数，$n$ 为判断结论"错误"的题目总数。

1. 二响应逻辑回归模型的优势为 $\exp(x\beta)$，其中 $x$ 为解释变量。
2. 算术题 $1+1$ 的结果是确定现象。
3. 二响应逻辑回归模型的得分函数是似然方程。
4. 对于二响应模型 $\mathbb{E}(Y|x)=h(Z(x)\beta)$ 有 $\text{var}(Y|x)=h(Z(x)\beta)(1-h(Z(x)\beta))$。
5. 二响应模型 $\mathbb{E}(Y|x)=h(Z(x)\beta)$ 的优势 $O(x)=\frac{h(z\beta)}{1-h(z\beta)}$。
6. 一个事件的优势和其概率相互唯一确定。
7. 对于二响应变量 $Y$ 和解释变量 $X$，可以用条件概率描述二者之间的关系。
8. 观测信息矩阵是似然函数的雅可比矩阵。
9. 用迭代公式
$$\hat{\beta}^{(k+1)}=\hat{\beta}^{(k)}+F_{\text{obs}}^{-1}\left(\hat{\beta}^{(k)}\right)\left(s\left(\hat{\beta}^{(k)}\right)\right)^{\text{T}},\ k\geqslant1,$$
可以得到二响应广义线性模型的模型参数的似然估计值，其中 $S$ 为得分函数，$F_{\text{obs}}$ 为观测信息矩阵。
10. 在二响应广义线性模型中，父模型的剩余偏差总是小于子模型的AIC值。
11. 若响应变量 $Y\sim B(1,h(Z(x)\beta))$，则响应变量和解释变量 $x$ 之间满足二响应广义线性模型。
12. 二响应广义线性模型的对数似然比统计量值的计算量比Wald统计量值的计算量大。
13. 在用同一响应函数探讨响应变量和分类解释变量间关系时，用 $\mathcal{F}$ 表示设计向量为解释变量的哑变量编码所对应的模型族，$\mathcal{G}$ 表示设计向量为解释变量的独热编码所对应的模型族，则 $\mathcal{F}$ 和 $\mathcal{G}$ 是相同的模型族。
14. 二响应广义线性模型的对数似然比统计量值的计算量比得分统计量值的计算量大。
15. 在应用同一响应函数探讨响应变量和两个分类解释变量间关系时，不同的设计向量对应着不同的模型族。
16. 在用R语言函数 glm() 拟合样本观测数据后，可以删除所有 p 值小于0.01的参数，重新建立最优子模型。
17. 在二响应广义线性模型中，父模型的剩余偏差总是小于子模型的剩余偏差。
18. 在二响应广义线性模型中，用皮尔逊统计量解答关于模型参数的假设检验问题。
19. 在响应函数相同的情况下，设计向量和模型参数唯一决定广义线性模型。
20. 能应用线性回归模型研究响应变量和解释变量之间关系的前提是：他们之间必须满足线性模型。
21. 对于二响应逻辑回归模型的模型参数假设检验问题 $H_0: C\beta=0$，可以用R语言函数 glm() 近似计算限制似然估计量的值。
22. 现实模型中的模型误差为随机变量。
23. 广义线性模型的模型参数永远不会改变。
24. 在探讨响应变量 $Y$ 和分类解释变量 $X$ 间关系时，只需将解释变量量化为数值变量 $w$，则应用广义线性模型建模族 $\mathbb{E}(Y|w)=h(a+bw)$ 中包含了拟合效果最佳的模型。
25. 对于二响应模型 $\mathbb{E}(Y|x)=h(Z(x)\beta)$，有 $Y\sim B(1,h(Z(x)\beta))$。
26. 似然方程的解是似然估计。
27. 用模型 $Y=Z(X)\beta$ 拟合样本观测数据意味着样本观测数据来自这一模型。
28. 条件期望模型 $Y=\mathbb{E}(Y|X)+\varepsilon$ 的应用障碍是无法确定 $X$ 的值。
29. 若二响应变量 $Y$ 是取值为1和2，解释变量为 $X$，则可以通过条件概率 $\mathbb{E}(Y|X)$ 建立描述二者关系的模型。
30. 在探讨响应变量和两个分类解释变量间关系时，可将两个解释变量的独热编码作为设计向量的子向量，用此设计向量对应的模型族拟合样本观测数据。

---

## 第二次考试（glinear20250422）

判断以下各题的陈述是否正确。分数计算公式为 $m-n$，其中 $m$ 为判断结论"正确"的题目总数，$n$ 为判断结论"错误"的题目总数。

1. 考虑二响应模型的模型参数第 $i$ 分量的假设检验问题 $H_0:\beta_i=0$，可以用统计量 $\frac{\hat{\beta}_i}{\sqrt{a_{ii}}}$ 构建计算 $p$ 值的公式，其中 $a_{ii}$ 为 $\left(F\left(\hat{\beta}\right)\right)^{-1}$ 的对角线上的第 $i$ 个元素。
2. 在研究二响应变量和通过调查问卷收集的解释变量间关系时，应该基于解释变量量化为效应编码或哑变量编码构建设计向量。
3. 在二响应广义线性模型中，可以通过经验预报公式 $\hat{Y}_c=1_{\{h(Z(x)\beta)>c\}}$ 预报相应变量的值，其中 $c$ 为阈值。阈值越大，TPR越小。
4. 考虑二响应广义线性模型的模型参数 $\beta$ 的假设检验问题
$$H_0:C\beta=\xi,$$
$\left(C\hat{\beta}-\xi\right)^{\mathrm{T}}\left(C\hat{\beta}-\xi\right)$ 越小越有利于原假设，其中 $\hat{\beta}$ 是似然估计量。
5. 对应任何设计向量 $z$ 和模型参数 $\beta$ 有 $\frac{\partial(z\beta)}{\partial\beta}(u)=z$。
6. 在二响应广义线性模型中，可以通过经验预报公式 $\hat{Y}_c=1_{\{h(Z(x)\beta)>c\}}$ 预报相应变量的值，其中 $c$ 为阈值。阈值越小，FPR越小。
7. 对于任何 $m\times p$ 矩阵 $z$ 和 $p$ 维列向量 $\beta$，有 $\frac{\partial(z\beta)}{\partial\beta}(u)=z$。
8. 对于二响应模型。如果 $\mathbb{P}(Y=1|X=x)>0.99$，则断言"解释变量值 $x$ 所对应的相应变量 $Y$ 的值为1"犯错误的概率为0.01。
9. 考虑二响应广义线性模型的模型参数 $\beta$ 的假设检验问题
$$H_0:C\beta=\xi,$$
$l\left(\tilde{\beta}\right)-l\left(\hat{\beta}\right)$ 越小越有利于原假设，其中 $\hat{\beta}$ 和 $\tilde{\beta}$ 分别是似然估计量和限制似然估计量。
10. 在用二响应模型拟合样本观测数据时，当原始样本观测数据严重违背经验知识时，可尝试通过经验样本数据改善样本观测数据的拟合效果。
11. ROC曲线是通过预报响应变量值为1的阈值来绘制。
12. 在二响应广义线性模型中，应该选择ROC曲线上离 $(0,1)$ 最近的点所对应的阈值构建响应变量的经验预报公式。
13. 在二响应广义线性模型中，可以通过经验预报公式 $\hat{Y}_c=1_{\{h(Z(x)\beta)>c\}}$ 预报相应变量的值，其中 $c$ 为阈值。阈值越小，TPR越小。
14. 若 $\hat{\beta}_i$ 是二响应模型的模型参数的似然估计量的第 $i$ 分量，则 $\frac{\hat{\beta}_i}{\sqrt{a_{ii}}}\stackrel{a}{\sim}N(0,1)$。
15. 在二响应广义线性模型中，可以通过经验预报公式 $\hat{Y}_c=1_{\{h(Z(x)\beta)>c\}}$ 预报相应变量的值，其中 $c$ 为阈值。阈值越大，FPR越小。
16. 在用逐步回归方法筛选最优子模型的过程中，可以用剩余偏差作为模型是否优秀的衡量指标。
17. 在二响应广义线性模型中，当模型参数 $\beta$ 满足条件 $C\beta=\xi$ 时，通常
$$s\left(\tilde{\beta}\right) F^{-1}\left(\tilde{\beta}\right)\left(s\left(\tilde{\beta}\right)\right)^{\mathrm{T}},$$
其中 $\tilde{\beta}$ 是限制似然估计量，$F\left(\tilde{\beta}\right)$ 是Fisher信息矩阵在限制似然估计量处的值，$r$ 是限制矩阵 $C$ 的秩。
18. 对于二响应模型。如果 $\mathbb{P}(Y=1|X=x)>0.99$，则解释变量值 $x$ 所对应的响应变量 $Y$ 的值是1。
19. 考虑二响应广义线性模型的模型参数 $\beta$ 的假设检验问题
$$H_0:C\beta=\xi,$$
$s\left(\tilde{\beta}\right)\left(F\left(\tilde{\beta}\right)\right)^{-1}\left(s\left(\tilde{\beta}\right)\right)^{\mathrm{T}}$ 越小越有利于原假设，其中 $\tilde{\beta}$ 是限制似然估计量，$F\left(\tilde{\beta}\right)$ 是Fisher信息矩阵在限制似然估计量处的值。
20. 考虑二响应模型的模型参数第 $i$ 分量的假设检验问题 $H_0:\beta_i=0$，则 $\frac{\hat{\beta}_i}{\sqrt{a_{ii}}}$ 越大，越有利于原假设，其中 $a_{ii}$ 为 $\left(F\left(\hat{\beta}\right)\right)^{-1}$ 的对角线上的第 $i$ 个元素。
21. 在二响应广义线性模型中，当模型参数 $\beta$ 满足条件 $C\beta=\xi$ 时，
$$\left(C\hat{\beta}-\xi\right)^{\mathrm{T}}\left(C\left(F\left(\hat{\beta}\right)\right)^{-1}C^{\mathrm{T}}\right)^{-1}\left(C\hat{\beta}-\xi\right)\stackrel{a}{\sim}\chi^{2}(r),$$
其中 $F\left(\hat{\beta}\right)$ 为Fisher信息矩阵在似然估计量处的值，$r$ 是限制矩阵 $C$ 的秩。
22. ROC曲线的坐标横轴为FPR，坐标纵轴为TPR。
23. 对于光滑 $m$ 维向量值多元函数 $f(x)=\left(f_1(x),\ldots,f_m(x)\right)^{\mathrm{T}}:\mathbb{R}^n\to\mathbb{R}^m$，有
$$\frac{\partial f}{\partial x}(u)=\left(\begin{array}{c}\frac{\partial f_1}{\partial x}(u)\\\vdots\\\frac{\partial f_m}{\partial x}(u)\end{array}\right)$$

---

## 第三次考试（glinear20250429）

判断以下各题的陈述是否正确。分数计算公式为 $m-n$，其中 $m$ 为判断结论"正确"的题目总数，$n$ 为判断结论"错误"的题目总数。

1. 在二响应广义线性模型中，$h(Z\beta)$ 是 $Y=1$ 的条件概率，其中 $h$ 是响应函数，$Z$ 是设计向量，$\beta$ 是模型参数。
2. 函数 anova() 是用于对数似然比检验的函数。
3. 对于二响应模型。如果 $\mathbb{P}(Y=1|X=x)>0.99$，则断言"解释变量值 $x$ 所对应的相应变量 $Y$ 的值为1"犯错误的概率小于0.01。
4. 在R语言程序代码 nnet::multinom(y~.,data=myData) 中，数据框 myData 中必须要有一列的名称为 y。
5. 在二响应广义线性模型中，可以用 $y=\mathbb{1}_{[0,h(Z(x)\beta]}(x)$ 模拟解释变量 $x$ 所对应的响应变量值，其中 $h$ 是响应函数，$Z(x)$ 是解释变量 $x$ 所决定的设计向量，$\beta$ 是模型参数。
6. 运行程序代码 myF<-function(){x<-1};x^2;y<-myF() 后，x 的值是1。
7. 在二响应广义线性模型中，当模型参数 $\beta$ 满足条件 $C\beta=\xi$ 时，通常
$$s\left(\bar{\beta}\right) F^{-1}\left(\bar{\beta}\right)\left(s\left(\bar{\beta}\right)\right)^{\mathrm{T}} \sim \chi^{2}(r),$$
其中 $\bar{\beta}$ 是限制似然估计量，$F\left(\bar{\beta}\right)$ 是Fisher信息矩阵在限制似然估计量处的值，$r$ 是限制矩阵 $C$ 的秩。
8. 若 $f:\mathbb{R}^k\to\mathbb{R}^m$ 和 $g:\mathbb{R}^m\to\mathbb{R}^n$ 都是向量值光滑函数，则
$$\frac{\partial g\circ f}{\partial x}(u)=\left(\frac{\partial g}{\partial y}(f(u))\right)\left(\frac{\partial f}{\partial x}(u)\right).$$
9. 在应用 $k$ 响应广义线性模型拟合样本观测数据时，程序代码 y~x1:x2 表示设计量为 $(1,\mathrm{x}1*\mathrm{x}2*\mathrm{x}3)$。
10. $k$ 响应广义线性模型可以用条件概率表示。
11. 函数 step() 是用于逐步回归分析的函数。
12. 若 $f:\mathbb{R}^k\to\mathbb{R}^m$ 和 $g:\mathbb{R}^m\to\mathbb{R}^n$ 都是向量值光滑函数，则
$$\frac{\partial g\circ f}{\partial x}(u)=\left(\frac{\partial f}{\partial x}(u)\right)\left(\frac{\partial g}{\partial y}(f(u))\right).$$
13. 在二响应广义线性模型中，可以用 $y=\mathbb{1}_{(0.5,1]}(h(Z(x)\beta))$ 模拟解释变量 $x$ 所对应的响应变量值，其中 $h$ 是响应函数，$Z(x)$ 是解释变量 $x$ 所决定的设计向量，$\beta$ 是模型参数。
14. R语言的神经网络包 nnet 包中的函数 multinom() 的功能是用多响应逻辑回归模型拟合样本观测数据。
15. 在二响应广义线性模型中，可以用 $y=\mathbb{1}_{[0,0.5]}(h(Z(x)\beta))$ 模拟解释变量 $x$ 所对应的响应变量值，其中 $h$ 是响应函数，$Z(x)$ 是解释变量 $x$ 所决定的设计向量，$\beta$ 是模型参数。
16. 在使用 nnet 包中的函数 multinom() 时，需要建立存放样本观测数据的数据框，在这个数据框中响应变量占 $k-1$ 列，其中 $k$ 是响应变量的值域中元素的个数。
17. $k$ 响应广义线性模型可以用条件分布函数表示。
18. $k$ 响应逻辑回归模型的响应函数的第 $r$ 分量为 $h_r(s_1,\dots,s_k)=\frac{\exp(\theta_r)}{1+\sum_{j=1}^k\exp(\theta_j)}$。
19. 在用 $k$ 响应广义线性模型拟合样本观测数据时，程序代码 $\text{y}\sim\text{x}1:\text{x}2$ 表示模型的设计向量为 $(1,x_1 x_2)$，其中 $\text{x}1$ 和 $\text{x}2$ 分别是 $x_1$ 和 $x_2$ 的样本观测数据。
20. 在 $k$ 响应广义线性模型 $\mathbb{E}(Y|X)=h(Z\beta)$ 中，响应变量 $Y$ 的条件方差矩阵为 $\text{diag}(h(Z\beta))$。
21. 当响应变量为分类变量时，可以用响应变量的哑变量编码的条件期望建立多响应广义线性模型。
22. 在二响应广义线性模型中，可以用 $y=\text{rbinom}(100,1,h(Z(x)\beta))$ 模拟解释变量 $x$ 所对应的100个响应变量值，其中 $h$ 是响应函数，$Z(x)$ 是解释变量 $x$ 所决定的设计向量，$\beta$ 是模型参数。
23. $k$ 响应广义线性模型的响应函数是 $\mathbb{R}^k$ 到 $\mathbb{R}^k$ 的映射。
24. 运行程序代码 $\text{myF}\leftarrow\text{function}()\{\text{x}\ll-1\};\text{x}\ll-2;\text{y}\ll\text{myF}()$ 后，$\text{x}$ 和 $\text{y}$ 的值相等。
25. $k$ 响应广义线性模型的响应函数是 $k$ 维向量值函数。
26. 在用二响应广义线性模型拟合样本观测矩阵时，程序代码 $\text{y}\sim\text{x}1*\text{x}2$ 表示设计向量为 $(1,x_1,x_2,x_1 x_2)$，其中 $\text{x}1$ 和 $\text{x}2$ 分别是 $x_1$ 和 $x_2$ 的样本观测数据。

---

## 第四次考试（glinear20250430）

判断以下各题的陈述是否正确。分数计算公式为 $m-n$，其中 $m$ 为判断结论"正确"的题目总数，$n$ 为判断结论"错误"的题目总数。

1. 考虑二响应广义线性模型的模型参数 $\beta$ 的假设检验问题
$$H_0:C\beta=\xi,$$
$l\left(\hat{\beta}\right)-l\left(\bar{\beta}\right)$ 越小越有利于原假设，其中 $\hat{\beta}$ 和 $\bar{\beta}$ 分别是似然估计量和限制似然估计量。
2. 在探讨响应变量和两个分类解释变量间关系时，可将两个解释变量的独热编码作为设计向量的子向量，用此设计向量对应的模型族拟合样本观测数据。
3. 现实模型中的模型误差为随机变量。
4. 对应任何设计向量 $z$ 和模型参数 $\beta$ 有 $\frac{\partial\left(z\beta\right)}{\partial\beta}(u)=z$。
5. 在用同一响应函数探讨响应变量和分类解释变量间关系时，用 $\mathcal{F}$ 表示设计向量为解释变量的哑变量编码所对应的模型族，$\mathcal{G}$ 表示设计向量为解释变量的效应编码所对应的模型族，则 $\mathcal{F}$ 和 $\mathcal{G}$ 是不同的模型族。
6. 二响应逻辑回归模型的优势为 $\exp(x\beta)$，其中 $x$ 为解释变量。
7. 统计学认为刻画未知现象的理想模型是客观存在的。
8. 似然方程的解是似然估计。
9. 条件期望模型 $Y=\mathbb{E}(Y|X)+\varepsilon$ 的应用障碍是无法确定 $X$ 的值。
10. 在应用同一响应函数探讨响应变量和两个分类解释变量间关系时，不同的设计向量对应着不同的模型族。
11. 在二响应广义线性模型中，当模型参数 $\beta$ 满足条件 $C\beta=\xi$ 时，
$$\left(C\hat{\beta}-\xi\right)^{\mathrm{T}}\left(C\left(F\left(\hat{\beta}\right)\right)^{-1}C^{\mathrm{T}}\right)^{-1}\left(C\hat{\beta}-\xi\right)\stackrel{\circ}{\sim}\chi^{2}(r),$$
其中 $F\left(\hat{\beta}\right)$ 为Fisher信息矩阵在似然估计量处的值，$r$ 是限制矩阵 $C$ 的秩。
12. 函数 anova() 是用于对数似然比检验的函数。
13. 用线性模型拟合样本观测数据时，必须保证模型误差为随机变量。
14. 若二响应变量 $Y$ 是取值为1和2，解释变量为 $X$，则可以通过条件概率 $\mathbb{E}(Y|X)$ 建立描述二者关系的模型。
15. 对于任何 $m\times p$ 矩阵 $z$ 和 $p$ 维列向量 $\beta$，有 $\frac{\partial\left(z\beta\right)}{\partial\beta}(u)=z$。
16. 若 $\hat{\beta}_i$ 是二响应模型的模型参数的似然估计量的第 $i$ 分量，则 $\frac{\hat{\beta}_i}{\sqrt{\hat{\sigma}_{ii}}}\stackrel{a}{\sim}N(0,1)$。
17. 对于二响应模型。如果 $\mathbb{P}(Y=1|X=x)>0.99$，则断言"解释变量值 $x$ 所对应的相应变量 $Y$ 的值为1"犯错误的概率为0.01。
18. 在用同一响应函数探讨响应变量和分类解释变量间关系时，用 $\mathcal{F}$ 表示设计向量为解释变量的哑变量编码所对应的模型族，$\mathcal{G}$ 表示设计向量为解释变量的效应编码所对应的模型族，则 $\mathcal{F}$ 和 $\mathcal{G}$ 是相同的模型族。
19. $k$ 响应广义线性模型的响应函数是 $k$ 维向量值函数。
20. $k$ 响应广义线性模型的响应函数是 $\mathbb{R}^k$ 到 $\mathbb{R}^k$ 的映射。
21. 运行程序代码 myF<-function(){x<<-1};x<-2;y<-myF() 后，x 和 y 的值相等。
22. 如果知道 $\mathbb{E}(Y|X)$，理想预报公式的预报误差就是0。
23. 称
$$Y=\mathbb{E}(Y|X)$$
为 $Y$ 的理想预报公式的原因之一是：在已知 $X$ 情况下用 $\mathbb{E}(Y|X)$ 预报 $Y$ 的误差最小。
24. 在用逐步回归方法筛选最优子模型的过程中，可以用剩余偏差作为模型是否优秀的衡量指标。
25. $k$ 响应逻辑回归模型的响应函数的第 $r$ 分量为 $h_r(s_1,\ldots,s_k)=\frac{\exp(s_r)}{1+\sum_{j=1}^k\exp(s_j)}$。
26. 广义线性模型的模型参数永远不会改变。
27. 用迭代公式
$$\hat{\beta}^{(k+1)}=\hat{\beta}^{(k)}+F_{\mathrm{obs}}^{-1}\left(\hat{\beta}^{(k)}\right)\left(s\left(\hat{\beta}^{(k)}\right)\right)^{\mathrm{T}},\ k\geqslant1,$$
可以得到二响应广义线性模型的模型参数的似然估计值，其中 $S$ 为得分函数，$F_{\mathrm{obs}}$ 为观测信息矩阵。
