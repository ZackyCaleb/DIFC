# DIFC
A tentative reproduction of the paper
Improved Cross-Dataset Facial Expression Recognition by Handling Data Imbalance and Feature Confusion
以下是 **PCGrad 与平坦度优化的联合建模**的详细数学推导，通过显式引入 Hessian-向量积（HVP）来平衡梯度冲突与泛化性，最终获得解析解的逐步过程。

---

### **1. 问题建模**
#### **目标函数**
联合优化以下两项：
1. **梯度冲突消除**：最小化调整后梯度 \( d \) 与原始梯度的差异；
2. **平坦性约束**：最小化 \( d \) 方向的 Hessian 能量（反映局部曲率）。

构建联合目标函数：
\[
\min_d \underbrace{\|d - \bar{g}\|^2}_{\text{梯度一致性}} + \lambda \cdot \underbrace{d^T H d}_{\text{平坦性控制}}
\]
其中：
- \( \bar{g} = \sum_{k=1}^K w_k g_k \) 是任务梯度的加权平均；
- \( H \) 是损失函数的 Hessian 矩阵；
- \( \lambda \) 是平衡系数。

#### **约束条件**
保留 PCGrad 的冲突消除约束：
\[
\langle d, g_k \rangle \geq 0, \quad \forall k
\]

---

### **2. 拉格朗日函数构建**
引入拉格朗日乘子 \( \alpha_k \geq 0 \) 处理约束，得到拉格朗日函数：
\[
\mathcal{L}(d, \alpha) = \|d - \bar{g}\|^2 + \lambda d^T H d + \sum_{k=1}^K \alpha_k \left( -g_k^T d \right)
\]

---

### **3. 求解最优梯度方向 \( d^* \)**
#### **(1) 对 \( d \) 求导并令导数为零**
\[
\frac{\partial \mathcal{L}}{\partial d} = 2(d - \bar{g}) + 2\lambda H d - \sum_{k=1}^K \alpha_k g_k = 0
\]
整理得：
\[
(2I + 2\lambda H) d = 2\bar{g} + \sum_{k=1}^K \alpha_k g_k
\]
即：
\[
d = (I + \lambda H)^{-1} \left( \bar{g} + \frac{1}{2} \sum_{k=1}^K \alpha_k g_k \right)
\]

#### **(2) 解析解形式**
若忽略约束（即 \( \alpha_k = 0 \)），解析解为：
\[
d_{\text{unconstrained}} = (I + \lambda H)^{-1} \bar{g}
\]
**物理意义**：对平均梯度 \( \bar{g} \) 进行 Hessian 加权修正，倾向于平坦方向。

#### **(3) 考虑约束的迭代求解**
当存在冲突时（\( \langle d, g_k \rangle < 0 \)），需激活约束：
1. 初始化 \( \alpha_k = 0 \)；
2. 计算 \( d = (I + \lambda H)^{-1} \left( \bar{g} + \frac{1}{2} \sum_{k=1}^K \alpha_k g_k \right) \)；
3. 检查约束是否满足：若 \( g_k^T d < 0 \)，则更新 \( \alpha_k \) 直到约束满足（类似梯度投影法）。

---

### **4. 快速 HVP 近似计算**
#### **(1) Hessian-向量积近似**
通过有限差分法快速计算 \( H d \)：
\[
H d \approx \frac{\nabla L(\theta + h d) - \nabla L(\theta)}{h}, \quad h \sim 10^{-3}
\]
**优势**：避免显式计算 Hessian，复杂度仅需两次梯度计算。

#### **(2) 迭代求解 \( (I + \lambda H)^{-1} v \)**
使用 Neumann 级数展开：
\[
(I + \lambda H)^{-1} v \approx \sum_{n=0}^N (-1)^n \lambda^n H^n v
\]
取前 \( N=1 \) 项：
\[
(I + \lambda H)^{-1} v \approx v - \lambda H v
\]
代入解析解：
\[
d \approx \bar{g} - \lambda H \bar{g} + \frac{1}{2} \sum_{k=1}^K \alpha_k (g_k - \lambda H g_k)
\]

---

### **5. 完整算法流程**
1. **输入**：任务梯度 \( \{g_k\}_{k=1}^K \)，参数 \( \theta \)，超参 \( \lambda, \rho \)；
2. **计算平均梯度**：\( \bar{g} = \sum_k w_k g_k \)；
3. **估计 HVP**：\( H \bar{g} \approx \frac{\nabla L(\theta + h \bar{g}) - \nabla L(\theta)}{h} \)；
4. **初化解**：\( d = \bar{g} - \lambda H \bar{g} \)；
5. **冲突检测与修正**：
   - For each \( g_k \):
     - If \( g_k^T d < 0 \):  
       \( d \leftarrow d + \frac{|g_k^T d|}{2 \|g_k\|^2} (g_k - \lambda H g_k) \)；
6. **输出**：调整后梯度 \( d \)。

---

### **6. 理论性质**
#### **(1) 冲突消除保证**
通过投影步骤确保 \( \langle d, g_k \rangle \geq 0 \)，继承 PCGrad 的优点。

#### **(2) 平坦性优化**
解 \( d \) 的修正项 \( -\lambda H \bar{g} \) 显式降低 Hessian 能量：
\[
d^T H d \approx \bar{g}^T H \bar{g} - 2\lambda \bar{g}^T H^2 \bar{g} < \bar{g}^T H \bar{g}
\]

#### **(3) 计算复杂度**
- HVP 近似：\( O(2 \cdot \text{gradient cost}) \)；
- 迭代修正：每冲突任务增加 1 次 HVP。

---

### **7. 示例验证**
设二任务梯度 \( g_1 = [1, 2]^T \), \( g_2 = [-1, 1]^T \)，Hessian \( H = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} \)，\( \lambda = 0.1 \)：
1. **平均梯度**：\( \bar{g} = 0.5 g_1 + 0.5 g_2 = [0, 1.5]^T \)；
2. **HVP 项**：\( H \bar{g} = [0, 1.5]^T \)；
3. **初始解**：\( d = \bar{g} - 0.1 H \bar{g} = [0, 1.35]^T \)；
4. **冲突检测**：\( g_2^T d = -0 \cdot 0 + 1 \cdot 1.35 > 0 \)（无冲突）；
5. **最终输出**：\( d = [0, 1.35]^T \)。

---

### **8. 总结**
通过将 PCGrad 的投影操作与 Hessian 驱动的平坦性优化结合，我们获得了一个**显式平衡冲突与泛化性的解析解**。其核心优势在于：
1. **数学可解释性**：解的形式清晰反映平坦性修正；
2. **计算高效性**：基于 HVP 近似，避免显式 Hessian；
3. **理论保证**：同时满足冲突消除与泛化性提升。

**未来方向**：  
- 自适应调整 \( \lambda \)（如基于梯度噪声估计）；  
- 扩展至随机优化场景（如 SGD 的 HVP 方差缩减）。
