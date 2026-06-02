# 双语阅读笔记 / Bilingual Reading Notes

---

## 论文信息 / Paper Information

| 项目 | 内容 |
|------|------|
| **标题** | Fast freeform optics design for high-contrast irradiance tailoring using optics-heuristic auction algorithm and rejection sampling |
| **中文译题** | 基于光学启发式拍卖算法与拒绝采样的高对比度辐照度调控快速自由曲面光学设计 |
| **作者** | Haoran Li, Haisong Tang, Zexin Feng (李浩然, 唐海松, 冯泽心) |
| **机构** | 北京理工大学 光电学院 / 混合现实与先进显示北京市工程研究中心 / 光电成像技术与系统教育部重点实验室 |
| **期刊** | Optics Express, Vol. 33, No. 26, 29 Dec 2025 |
| **出版方** | Optica Publishing Group (Open Access) |
| **通讯作者** | fzx84@126.com |

---

## Abstract / 摘要

> **原文:**
> Freeform lens design for producing a high-contrast irradiance distribution from a zero-étendue source can be related to an optimal transport problem. This problem can be discretized into a linear assignment problem, but existing solutions are often computationally expensive and time-consuming. To accelerate the process, we introduce rejection sampling for faster discretization and a heuristic auction algorithm with an optimization step size adapted to optical features enabling the rapid solution of the linear assignment problem. Furthermore, freeform surface construction is automatically calculated to ensure surface continuity. The effectiveness of the proposed method is demonstrated through four examples designed to generate near-field and far-field distributions. These examples involve generating distributions in the shape of zero-background letters "BIT" and an Einstein portrait, from either parallel beams or point sources. In these examples, our method exhibits improved optical performance while significantly increasing the overall computational efficiency.

> **中文翻译:**
> 从零扩展量光源产生高对比度辐照度分布的自由曲面透镜设计可归结为最优传输问题。该问题可离散化为线性分配（LA）问题，但现有算法通常计算成本高、耗时长。为加速求解过程，本文引入拒绝采样以实现更快的离散化，并提出一种启发式拍卖算法——其优化步长根据光学特征自适应调整，从而快速求解线性分配问题。此外，自由曲面构建可根据连续性要求自动完成。通过四个生成近场和远场分布的算例验证了方法的有效性：分别在平行光束和点光源条件下生成零背景"BIT"字母和爱因斯坦肖像分布。结果表明，所提方法在光学性能提升的同时，显著提高了整体计算效率。

---

## 1. Introduction / 引言

辐照度调控（irradiance tailoring）是非成像光学中的典型逆问题，广泛应用于汽车照明、道路照明和建筑照明等领域。自由曲面光学突破了传统球面和非球面光学元件的限制，能够实现更灵活、更复杂的光分布。然而，自由曲面的高设计自由度使其设计极具挑战性。

在零扩展量假设下（即点光源或平行光束），自由曲面光学设计在大多数情况下可表示为 Monge-Ampère (MA) 类型的二阶非线性偏微分方程。Wu 等人推导了自由曲面透镜设计的 MA 方程并用 Newton 法数值求解 [6]。Ries 等人建立了不同的方程系统并求解了相应的等效非线性 PDE [5]。Boonkkamp 等人针对三种不同场景推导了三组方程并用迭代最小二乘法求解 [7]。

光线映射（ray mapping）方法也是常用手段，将设计分解为两步：计算合适的光线映射 + 重建曲面 [10-16]。虽然概念直观，但核心挑战在于计算可积映射，为此发展了迭代波前裁剪 [17] 和辛变换 [18] 等专门方法。然而，这些方法在高对比度辐照度调控中面临困难——源域或目标域的某些区域可能**能量为零**。

对于高对比度调控，经典支撑二次曲面法可框架化为具有特定代价函数的最优传输问题 [19-28]。Bykov 等人 [29] 采用基于 Voronoi 剖分的加权 Lloyd 算法进行离散化，并用拍卖算法求解离散最优传输问题（即 LA 问题），通过多尺度方法加速整个迭代过程。该方法在生成高对比度分布方面展示了显著的通用性，但其**瓶颈在于 Voronoi 采样的高计算成本和拍卖算法的耗时**。

**本文贡献：**
1. 提出融合光学特征的**启发式 ε-缩放 aggressive 拍卖算法**，根据光程特征调整优化步长
2. 使用**拒绝采样**替代 Voronoi 采样以加速离散化
3. 根据连续性要求**自动计算**自由曲面构建（基于 sympy 自动推导 B 样条基函数）

---

## 2. Principle / 原理

### 2.1 准直光束下的最优传输问题

自由曲面透镜由平面入射面和自由曲面出射面组成。准直光束沿 z 轴正方向传播，光线从源点 $(x, y, 0)$ 射入，与自由曲面交于 $(x, y, z(x, y))$。

源域和自由曲面分别由 $(x, y) \in S \subset \mathbb{R}^2$ 和 $(x, y) \in F \subset \mathbb{R}^2$ 参数化，其辐照度分布分别为 $E_S(x, y)$ 和 $E_F(x, y)$。目标面由 $(u, v) \in T \subset \mathbb{R}^2$ 描述，z 坐标为 $h(u, v)$，辐照度为 $E_T(u, v)$。

**连续形式的最优传输代价函数 (Eq. 1):**

$$C(M) = \iint d(x, y, M(x, y)) \, E_F(x, y) \sqrt{1 + \left[\frac{\partial z}{\partial x}\right]^2 + \left[\frac{\partial z}{\partial y}\right]^2} \, dx\,dy$$

其中 $d(x, y, M(x, y)) = \| ((x, y), z(x, y)) - (M(x, y), h(M(x, y))) \|_2$ 表示单根光线从自由曲面到目标面的**光程长度**。

**能量守恒约束 (Eq. 2):**

$$\iint_{\omega_F} E_F(x, y) \sqrt{1 + \left[\frac{\partial z}{\partial x}\right]^2 + \left[\frac{\partial z}{\partial y}\right]^2} \, dx\,dy = \iint_{M(\omega_F)} E_T(u, v) \, du\,dv$$

利用 $E_S = E_F \sqrt{1 + z_x^2 + z_y^2}$（光束垂直入射平面），方程简化为：

$$\begin{cases} M \in \{\arg\min C(M), \arg\max C(M)\} \\ C(M) = \iint d(x, y, M(x, y)) E_S(x, y) \, dx\,dy \\ \text{s.t. } \iint E_S(x, y) dxdy = \iint_{M(\omega_F)} E_T(u, v) dudv \end{cases}$$

> **关键点：** 对于高对比度分布（$E_T(u, v) = 0$ 的子区域），程函函数是 Lipschitz 连续的，几乎处处可微——不可微点集为 Lebesgue 零测集，因此**不影响最优传输的构建**（详见[21] Theorem 4.1 和[33] Theorem 2）。

确定映射 $M$ 后，折射光线单位方向向量 $\mathbf{R}$ 由式(4)给出，法向量 $\mathbf{N}$ 由 Snell 定律 (Eq. 5) 计算：

$$n_{in}(\mathbf{I} \times \mathbf{N}) = n_{out}(\mathbf{R} \times \mathbf{N})$$

然后用最小二乘法基于 $\mathbf{N}$ 构建自由曲面。

### 2.2 离散化为线性分配问题

将域 $F$ 和 $T$ 分别离散化为 $n$ 个等能量 $e$ 的子区域 $f_i$ 和 $t_j$，由特征点 $(x_i, y_i)$ 和 $(u_j, v_j)$ 表示：

$$\iint_{f_i} E_S(x, y) dxdy = \iint_{t_j} E_T(u, v) dudv = e$$

**离散代价函数 (Eq. 7):**

$$C_d(M) = \sum_{i=1}^{n} d(x_i, y_i, M(x_i, y_i))$$

目标：找到 $f_i$ 到 $t_j$ 的满射使代价函数最大化或最小化 → 典型的 LA 问题。常用 Hungarian 算法 [35] 和拍卖算法 [36]。

**Algorithm 1（迭代自由曲面构建）**：初始化矢高 $z^0$ → 每轮计算映射 $M$ → 更新矢高 $z^{k+1}$（固定点 $(x_{fix}, y_{fix}, z_{fix})$）→ 迭代至收敛（$\Delta z < tol$ 或 $k = k_{max}$）。

### 2.3 点光源情形的推广

点光源由强度分布 $I(\theta, \varphi)$ 描述，向 $z > 0$ 半空间发射。自由曲面由角度坐标 $(\theta, \varphi)$ 参数化，距离函数 $R_2(\theta, \varphi)$ 表示从原点到曲面的距离。

- 微分能量元用立体角表示 (Eq. 8): $I(\theta, \varphi) \sin\theta \, d\theta d\varphi$
- 简化方程 (Eq. 9): 用 $I(\theta, \varphi)$ 代替 $E_S$
- 坐标关系 (Eq. 10): $x = R_2 \sin\theta \cos\varphi$, $y = R_2 \sin\theta \sin\varphi$, $z = R_2 \cos\theta$

离散化后每个区域能量相等 (Eq. 11)，代价函数 (Eq. 12)：$C_d(M) = \sum d(\theta_i, \varphi_i, M(\theta_i, \varphi_i))$。固定点条件：$R_2(\theta_{fix}, \varphi_{fix}) = R_{fix}$。

---

## 3. Method / 方法

### 3.1 拒绝采样 / Rejection Sampling

**原问题：** 按能量分布将域 $T$ 严格离散化为 $n$ 个等能点是一个约束最优传输问题（Voronoi 剖分），求解极为复杂且耗时。

**拒绝采样方案：**
- Monte Carlo 方法：从均匀分布采样候选点，按 $E_T / \max(E_T)$ 概率接受
- 可并行生成，接受比例 $\approx \iint E_T \,/ \iint \max(E_T)$
- 可结合 Sobol 序列或 Fibonacci 序列提升效率
- 准直光束在源域 $S$ 采样，点光源在球面入射面采样

### 3.2 启发式拍卖算法 / Heuristic Auction Algorithm

**传统拍卖算法：**
- $n$ 个"人" $f_i$ 匹配 $n$ 个"物品" $t_j$，最大化总收益
- $c_{ij}$ = 光程长度 $d(x_i, y_i, u_j, v_j)$（即匹配收益）
- $p_j$ = 物品价格，$a_{ij} = c_{ij} - p_j$ = 净收益

**保守拍卖算法：** 每轮每人选择净收益最大的物品，冲突时竞价抬价 → 收敛慢（局部光程结构相似时难以区分）。

**Aggressive ε-缩放拍卖算法：**
- 引入 ε-CS 条件：允许在 $[a_{max} - \epsilon, a_{max}]$ 范围内选择，减少局部价格竞争
- $\epsilon = \alpha^L \epsilon_{start}$ ($0 < \alpha < 1$)：逐步精细化

**本文核心创新——启发式变体 (Fig. 5)：**

| | 传统 ε-缩放 | 启发式 ε-缩放 |
|------|------------|------------|
| ε 初始值 | 每轮相同 | 随曲面迭代递减 |
| ε 停止阈值 | 每轮相同 | 随曲面迭代递减 |
| 价格初始化 | 每轮从零开始 | 继承上一轮价格（热启动） |

$$\epsilon_{start}^{k+1} = \gamma \epsilon_{end}^{k} < \epsilon_{start}^{k}, \quad \gamma > 1$$

> **核心洞察：** 曲面构建初期，自由曲面尚不准确，收益 $c_{ij}$ 也不精确——此时不需要高精度求解 LA 问题。随着曲面迭代收敛，逐步提高 LA 求解精度。

**整个迭代过程可视为一个完整的 ε-缩放拍卖**，其中 $c_{ij}$ 在曲面构建后被更新。

**Algorithm 2:** 启发式 ε-缩放 aggressive 拍卖算法流程（详见原文）。

### 3.3 基于连续性的曲面构建

- 使用 **B 样条曲面** 表示自由曲面（良好的局部控制能力）
- 基于 Python **sympy** 库自动推导不同次数的基函数
- 可根据连续性要求灵活调整 B 样条次数（$C^m$ 连续 → 构建 $m$ 次 B 样条）
- **权衡：** 高次 B 样条提高可制造性，但计算时间增加；低次 B 样条可能导致光程计算不准确

---

## 4. Results / 实验结果

### 实验配置

| 参数 | 设定 |
|------|------|
| 透镜折射率 | 1.4936 |
| 评价指标 | RRMSE（相对均方根误差）|
| 渲染引擎 | Mitsuba 3，512×512 像素 |
| 硬件 | Intel i9-12900K CPU, 128 GB RAM |
| 算例 1, 2 目标距离 | 200 mm，尺寸 400×400 mm |
| 光源 | 4 mm 方形平行光束 |

---

### Example 1: 远场 "BIT" 字母（准直光束）

**离散化：** 目标分布被离散为约 900×900 个点。中心厚度固定 2.0 mm，最多 6 轮迭代。

| 方法 | RRMSE | 总计算时间 | 相对加速 |
|------|-------|-----------|---------|
| Lloyd + 传统拍卖 | 0.2942 | 18343.2 s | 基准 |
| Lloyd + 启发式拍卖 | ~0.27 | ~10042 s | ~1.8× |
| Rejection + 传统拍卖 | 0.2704 | 7200.2 s | ~2.5× |
| **Rejection + 启发式拍卖** | **0.2463** | **2507.0 s** | **~7.3×** |

**离散化阶段对比：** 拒绝采样仅需 **0.5 s**，加权 Lloyd 算法需 **8301.0 s**（约 16000 倍加速）。

**B 样条次数的影响：**

| B 样条次数 | 连续性 | RRMSE | 表面构建时间 |
|-----------|--------|-------|------------|
| 1 次 | C1 | 0.2529 | 101.3 s |
| 2 次 | C2 | 0.2463 | 463.1 s |
| 3 次 | C3 | 0.2452 | 2612.0 s |

---

### Example 2: 远场爱因斯坦肖像（准直光束）

| 方法 | RRMSE | 总计算时间 | 相对加速 |
|------|-------|-----------|---------|
| Lloyd + 传统拍卖 | 0.1776 | 22109.0 s | 基准 |
| Lloyd + 启发式拍卖 | ~0.135 | ~14455.8 s | ~1.5× |
| Rejection + 传统拍卖 | ~0.166 | ~10856.1 s | ~2.0× |
| **Rejection + 启发式拍卖** | **0.1342** | **1936.9 s** | **~11.4×** |

> 爱因斯坦肖像分布包含更多精细细节，本文方法相比传统方法提升更加显著（11.4× 加速）。

---

### Example 3: 多场景 "BIT" 分布

三种不同光源和靶面配置：

| 场景 | 光源 | 分布类型 | 靶面尺寸 | 距离 | RRMSE |
|------|------|---------|---------|------|-------|
| (a) | 4 mm 方孔准直光束 | 近场 | 8×8 mm | 6 mm | 0.2389 |
| (b) | 点光源（半球发射） | 远场 | 600×600 mm | 200 mm | 0.2729 |
| (c) | 点光源（60°半发散角） | 近场 | 8×8 mm | 4 mm | 0.2864 |

- 近场场景 (a) 最小化代价函数 → 凹透镜设计，同时减小矢高和抑制全内反射
- 点光源条件 RRMSE 略高于准直光束 → 高对比度区域的急剧弯曲在半球面上更难表示
- (c) 使用 60°限制而非半球发射以抑制全内反射

---

### Example 4: 多场景爱因斯坦肖像

| 场景 | 光源 | 分布类型 | RRMSE |
|------|------|---------|-------|
| (a) | 准直光束 | 近场（8×8mm@6mm） | 0.1265 |
| (b) | 点光源 | 远场（600×600mm@200mm） | 0.1360 |
| (c) | 点光源（全半球发射） | 近场（8×8mm@4mm） | 0.1218 |

> 三种配置光学性能相近，展示了算法的通用性。
> 场景 (c) 最耗时（5347.9 s），因半球发射源导致迭代中矢高大范围变化。

---

## 5. Conclusion / 结论

1. 提出了**启发式 ε-缩放 aggressive 拍卖算法**，将优化步长 ε 按光程特征在曲面构建迭代中动态缩放
2. 使用**拒绝采样**加速离散化过程
3. 根据**连续性要求**使用 B 样条自动构建自由曲面
4. 相比传统方法（Lloyd + 传统 ε-缩放拍卖），**计算效率提升 7~11 倍**，同时光学性能更好
5. 方法具有通用性：适用于准直光束/点光源 × 近场/远场不同组合
6. **未来方向：** 扩展到扩展光源的辐照度调控和亮度调节

---

## 方法对比总结 / Method Comparison Summary

| 环节 | 传统方法 | 本文方法 | 提速 |
|------|---------|---------|------|
| 离散化 | 加权 Lloyd 算法 (~8300 s) | 拒绝采样 (~0.5 s) | ~16000× |
| LA 求解 | 传统 ε-缩放拍卖 | 启发式 ε-缩放拍卖 | ~3.4× |
| 曲面构建 | 固定次数 B 样条 | sympy 自动推导基函数 | 灵活可调 |
| **总体** | 18343~22109 s | **1937~2507 s** | **7~11×** |

---

## 参考文献索引 / Key References

| 编号 | 作者 | 年份 | 核心贡献 |
|------|------|------|---------|
| [5] | Ries & Muschaweck | 2002 | 定制自由曲面光学表面的 PDE 方法 |
| [6] | Wu et al. | 2013 | 椭圆 MA 方程的非线性边界问题 |
| [7] | Boonkkamp et al. | 2025 | Hamilton 理论+最优传输+最小二乘的自由曲面逆设计 |
| [21] | Glimm & Oliker | 2003 | 单反射器系统与 Monge-Kantorovich 质量传输 |
| [29] | Bykov et al. | 2018 | **LA 问题用于自由曲面折射元件设计** |
| [30] | Bykov et al. | 2020 | **多尺度方法 + LA 问题加速** |
| [31] | Flury | 1990 | 接受-拒绝采样方法 |
| [36] | Bertsekas | 1998 | 网络优化：连续与离散模型（拍卖算法理论基础） |
| [38] | Bertsekas | 2024 | 新拍卖算法用于分配问题及其扩展 |
| [40] | Jakob et al. | 2022 | Mitsuba 3: 可微渲染 JIT 编译器 |

---

*笔记生成时间：2026-05-29 | 使用 MinerU (flash-extract) 提取原文 + Claude Code 双语翻译整理*
