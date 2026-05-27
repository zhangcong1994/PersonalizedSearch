# exp-006 实验结果与分析

> 实验目标：在 exp-003 S4 粗排基础上，通过 DeepSeek gap analysis 分析第一轮检索缺口、生成 1 条改写查询，经 Round 2 Dense 检索后与 Round 1 做平权 RRF 融合，提升 Recall@50 和 Hit@50
>
> 实验类型：两轮检索管线——Round 1 复用 exp-003 S4 结果（零新增检索成本），Gap 分析调 DeepSeek API，Round 2 现场 Dense 检索
>
> 核心指标：Recall@50（主指标）、Hit@50（辅助）
>
> 实验阶段：Phase 1（200 条 smoke test）→ Phase 2（912 条全量测试集）

---

## 数据集与配置

| 项目 | 数值 |
|------|------|
| 数据集 | T2Ranking dev (test split, seed=42) |
| Phase 1 样本 | 200 条 |
| Phase 2 样本 | 912 条（全量测试集） |
| Round 1 | exp-003 S4: D-B0 + D-P2 + D-HyDE + B-B0 → RRF k=60 → top-50 |
| Gap 分析模型 | DeepSeek-chat（temperature=0.1, max_tokens=1024） |
| Gap 分析输入 | 原查询 + Round 1 Top-10 passage（每段 ≤400 字） |
| Gap 分析输出 | diagnosis + 1 条 reformulated_query + negative_signals |
| Round 2 检索 | Dense only（m3e-base, 768 维, ChromaDB） |
| Round 2 每 query 取 top-50 |  |
| RRF 融合 | Round 1 4 路 + Round 2 1 路，RRF k=60，取 top-50 |

---

## 一、Phase 2 最终结果（912 条全量测试集）

| 指标 | Round 1 Only | Round 1+2 | Δ 绝对 | Δ 相对 |
|------|-------------|-----------|--------|--------|
| **Recall@50** | 0.6912 | **0.6989** | **+0.0077** | **+1.1%** |
| **Hit@50** | 0.8991 | **0.9002** | **+0.0011** | **+0.1%** |

- Avg Round 2 routes: 1.0（每条查询生成 1 条改写）
- API 调用：912 次，总耗时 445.0s（488ms/q）
- 成本：约 ¥4.5

### 对比：Phase 1 vs Phase 2

| 指标 | Phase 1（200 条） | Phase 2（912 条） |
|------|:---:|:---:|
| Round 1 R@50 | 0.6778 | 0.6912 |
| Round 1+2 R@50 | 0.6937 | 0.6989 |
| Δ Recall@50 | **+1.59pp** | **+0.77pp** |
| Δ Hit@50 | **+0.50pp** | **+0.11pp** |

Phase 2 的 Δ 约为 Phase 1 的一半——200 条小样本放大了个别好 case 的贡献，912 条回归均值。方向仍然为正，但增益比 Phase 1 预期的更小。

---

## 二、关键中间发现：跨模型互补的伪增益

在执行过程中，发现了一组对比数据：

| Round 2 检索模型 | Recall@50 Δ (200 条) | 解释 |
|:---|:---|:---|
| bge-small-zh-v1.5（512 维） | **+3.01pp** | 含跨模型互补伪增益 |
| m3e-base（768 维，与 Round 1 一致） | **+1.59pp** | 纯改写增益（200 条小样本） |
| m3e-base（768 维，全量 912 条） | **+0.77pp** | 纯改写增益（最终结果） |

### 为什么 bge-small 的 Δ 更大？

bge-small 与 m3e-base 的 embedding 空间不对齐。Round 1 的三路 Dense 路由在 m3e-base 空间中已覆盖了大部分相关 passage，但 bge-small 空间中存在与 m3e-base **互补的"盲区"**——bge-small 检索到的 passage 恰好是 Round 1 未覆盖的新区域。

这说明**在实验中不能混用不同 embedding 模型做 RRF 融合**——不同模型空间的排名互补会制造伪增益，污染实验结论。

---

## 三、增益分析

### 3.1 全量 912 条：+0.77pp 意味着什么

- Recall@50 从 0.6912 → 0.6989，绝对值提升不到 1 个点
- 912 条中多命中了 **约 7 条** query 中的额外相关 passage
- Hit@50 从 0.8991 → 0.9002，仅多救了 **1 条** zero-hit query

### 3.2 增益为什么比预期小

**Round 1 的 3 路 Dense 路由已经几乎填满了这个 embedding 空间的天花板。** D-P2 是领域 few-shot 盲写、D-HyDE 是伪答案检索，两个 LLM 驱动的路由已经覆盖了 D-B0 原查询在这个空间里搜不到的 passage。gap analysis 改写虽然在"看结果改写"模式上与 D-P2 的"盲写"不同，但 m3e-base 空间内真正未覆盖的相关 passage 已经很少了。

> 核心洞察：**在这个实验设置下，多轮检索 + gap analysis 改写对 Recall@50 的提升上界约为 +0.8pp。** 这不是策略无效，而是 Round 1 基线本身已经逼近了这个 embedding 空间的天花板。

### 3.3 Hit@50 几乎不变的原因

912 条中 zero-hit 约 92 条（10.1%）。其中：
- E 类（qrels 标注噪声）：~4 条，无法挽回
- A 类（pid mismatch / RRF 相关但 pid 不在 top-50）：~12 条，已在前 50 附近，RRF 融合难以再推动跨越阈值
- B/C/D 类（可改写型）：~15 条

理论上可救的 ~15 条中只救了 1 条。核心原因是**单条改写查询能引入的新 passage 和 Round 1 top-50 的重叠度很高**——改写后的 query 在同一个 m3e-base 空间里搜到的 top passage，很大程度上已经被 Round 1 的三路路由覆盖了。

---

## 四、与 exp-003 的对照

| 对比维度 | exp-003 | exp-006 |
|----------|---------|---------|
| 查询改写方式 | D-P2: few-shot 盲写（不看检索结果） | gap analysis: 看到 top-10 结果后改写 |
| 改写增益（vs 对应基线） | +3.6pp（D-P2 vs D-B0, 单路 Dense） | +0.77pp（Round 1+2 vs Round 1, 5 路 RRF） |
| 增益对手 | 单路 D-B0（R@50=0.6089） | 4 路 RRF（R@50=0.6912，含 D-P2 + D-HyDE） |

**D-P2 的 +3.6pp 是在单路上打原查询，而我们的 +0.77pp 是在 4 路融合上打 4 路融合**——对手不同，增益不可直接比。4 路融合的基线已经逼近 embedding 空间上界，gap analysis 所余的提升空间本来就很小。

---

## 五、结论

### 5.1 实验结论

- ✅ gap analysis + 改写查询的策略方向正确，全量 Recall@50 正向提升 **+0.77pp**
- ✅ "+1.59pp（小样本）→ +0.77pp（全量）" 的稀释符合统计预期
- ⚠️ **增益 ≤ 1pp，边际收益不足以作为独立创新点**

### 5.2 为什么这场实验值得做

虽然数据不够漂亮，但提供了一个**重要的归因结论**：

> 在强多路 Dense 融合（≥3 路，含 LLM 改写和 HyDE）的基线上，同一 embedding 空间内的 query 改写已经很难带来显著增益。真正的瓶颈是 **embedding 空间的信息表达上限**，而非 query 的表述角度。

这为下一步方向提供了清晰的指引：**如果想继续推高 Recall@50，需要突破 embedding 空间本身（换更强的模型、做微调），或者从检索范式上做改变（PRF、稀疏-稠密混合、迭代式多空间检索）。**

### 5.3 路线图更新

| 方向 | 优先级 | 理由 |
|------|:---:|------|
| **embedding 微调（exp-007）** | 🔴 高 | 嵌入空间是效率天花板，微调是直接突破方式 |
| PRF / 伪相关反馈 | 🟡 中 | 零 API 成本，可能带来 1-3pp |
| 多条改写查询（2-3 条） | 🟢 低 | 同空间内，多改写增益也很有限 |
| 换大 embedding 模型 | 🟢 低 | 建索引成本高，不如微调精细 |

---

## 六、下一步

- [x] Phase 2：全量测试集（912 条）跑最终指标
- [ ] 分析增益集中在哪些 query 类型（从 gap analysis 缓存的 diagnosis 中统计）
- [ ] 消融：gap analysis 中 thinking mode 的贡献（thinking vs non-thinking）
- [ ] 转向 exp-007：embedding 微调
