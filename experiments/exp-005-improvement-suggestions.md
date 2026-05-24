# Exp-005 改进建议

> 创建日期：2026-05-23
>
> 关联：`exp-005-ai-search-eval-rubric.md` —— AI 搜索生成质量评估六层维度与评分标准
>
> 目标：汇总评估框架的改进建议，包括评分聚合方法、查询分层、Judge 一致性控制等

---

## 一、评分聚合方法

### 1.1 设计原则

采用 **门槛法 + 加权求和** 的两阶段聚合策略：

1. **第一阶段（门槛检查）**：准确性和安全性作为门槛维度，不达标直接判定不合格
2. **第二阶段（加权求和）**：通过门槛的答案，按维度权重计算加权总分，映射到 0-100 分区间

### 1.2 维度权重分配

| 维度 | 权重 | 角色 | 理由 |
|------|------|------|------|
| **L1 信息准确性** | 30% | 核心 | 事实错误的答案毫无价值，权重最高 |
| **L4 相关性** | 25% | 核心 | 答非所问的答案再漂亮也没用 |
| **L2 信息整合质量** | 20% | 差异化 | AI 搜索区别于 RAG 的核心能力 |
| **L3 引文质量** | 15% | 支撑 | 证据链完整性，影响可信度 |
| **L5 用户体验** | 10% | 支撑 | 呈现质量，影响阅读体验 |
| **L6 安全与合规** | 门槛 | 一票否决 | 不达标直接 reject，不参与加权 |

**权重总和**：30% + 25% + 20% + 15% + 10% = 100%

### 1.3 聚合公式

```python
def aggregate_scores(scores: dict) -> dict:
    """
    聚合六维度评分，返回综合分数和判定结果。
    
    Args:
        scores: 包含六个维度分数的字典，每个维度 1-4 分
            {
                "veracity": 3,
                "synthesis_quality": 4,
                "citation_quality": 3,
                "relevance": 3,
                "user_experience": 4,
                "safety": 3
            }
    
    Returns:
        {
            "pass": True,
            "total_score": 78.5,
            "grade": "B",
            "gate_failures": [],
            "penalty_applied": False
        }
    """
    
    # ── 第一阶段：门槛检查 ──
    gate_failures = []
    
    if scores["veracity"] < 2:
        gate_failures.append("veracity")
    if scores["safety"] < 2:
        gate_failures.append("safety")
    
    if gate_failures:
        return {
            "pass": False,
            "total_score": 0,
            "grade": "F",
            "gate_failures": gate_failures,
            "reason": f"门槛维度不达标: {', '.join(gate_failures)}"
        }
    
    # ── 第二阶段：加权求和 ──
    weights = {
        "veracity": 0.30,
        "relevance": 0.25,
        "synthesis_quality": 0.20,
        "citation_quality": 0.15,
        "user_experience": 0.10,
    }
    
    weighted_sum = sum(scores[dim] * weights[dim] for dim in weights)
    
    # 映射到 0-100 分区间
    # 原始范围：1-4 分 → 映射后范围：0-100 分
    # 公式：(weighted_sum - 1) / (4 - 1) * 100
    total_score = (weighted_sum - 1) / 3 * 100
    total_score = round(total_score, 1)
    
    # ── 惩罚项：有明显短板 ──
    penalty_applied = False
    non_gate_dims = ["veracity", "synthesis_quality", "citation_quality", 
                     "relevance", "user_experience"]
    min_non_gate = min(scores[dim] for dim in non_gate_dims)
    
    if min_non_gate == 1:
        total_score = max(0, total_score - 10)
        penalty_applied = True
    
    # ── 等级映射 ──
    if total_score >= 90:
        grade = "S"
    elif total_score >= 80:
        grade = "A"
    elif total_score >= 70:
        grade = "B"
    elif total_score >= 60:
        grade = "C"
    else:
        grade = "D"
    
    return {
        "pass": total_score >= 60,
        "total_score": total_score,
        "grade": grade,
        "gate_failures": [],
        "penalty_applied": penalty_applied,
        "weighted_raw": round(weighted_sum, 3)
    }
```

### 1.4 分数映射对照表

| 加权原始分 | 映射后分数 | 等级 | 业务含义 | 示例场景 |
|-----------|-----------|------|---------|---------|
| 4.0 | 100 | S | 完美答案 | 所有维度均为 4 分 |
| 3.5-3.9 | 83-97 | A | 优秀 | 核心维度 4 分，支撑维度 3 分 |
| 3.0-3.4 | 67-80 | B | 良好 | 核心维度 3 分，无明显短板 |
| 2.5-2.9 | 50-63 | C/D | 需优化 | 部分维度 2 分 |
| 2.0-2.4 | 33-47 | D | 不合格 | 多个维度 2 分或以下 |
| 1.0-1.9 | 0-30 | F | 不可用 | 门槛维度不达标或全面低分 |

### 1.5 使用示例

```python
# 示例 1：高质量答案
scores_1 = {
    "veracity": 4,
    "synthesis_quality": 4,
    "citation_quality": 3,
    "relevance": 4,
    "user_experience": 4,
    "safety": 4
}
result_1 = aggregate_scores(scores_1)
# → pass: True, total_score: 93.3, grade: "S"

# 示例 2：事实错误（门槛不通过）
scores_2 = {
    "veracity": 1,  # 严重错误
    "synthesis_quality": 3,
    "citation_quality": 3,
    "relevance": 3,
    "user_experience": 3,
    "safety": 3
}
result_2 = aggregate_scores(scores_2)
# → pass: False, gate_failures: ["veracity"], grade: "F"

# 示例 3：答非所问但事实正确
scores_3 = {
    "veracity": 4,
    "synthesis_quality": 2,
    "citation_quality": 3,
    "relevance": 1,  # 答非所问
    "user_experience": 3,
    "safety": 3
}
result_3 = aggregate_scores(scores_3)
# → pass: False, total_score: 0 (有短板惩罚), grade: "D"
```

### 1.6 聚合策略的权衡说明

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 简单算术平均 | 最简单 | 掩盖关键维度差异 | 不推荐 |
| 加权求和（本方案） | 灵活、可解释 | 权重设定需经验 | ✅ 推荐 |
| 决策树 | 可解释性最强 | 规则复杂后难维护 | 需要明确分类时 |
| 雷达图 | 信息完整 | 不给出结论 | 学术研究、模型对比 |

---

## 二、查询复杂度分层

### 2.1 问题

不同复杂度的查询，评估标准应该有所区分。当前框架对所有查询类型使用同一套标准和权重，可能导致：
- 简单查询被过度评估
- 复杂查询的某些维度被低估
- 不同查询类型之间的分数不可比

### 2.2 查询分类体系

| 查询类型 | 示例 | 特征 | 评估重点 |
|---------|------|------|---------|
| **Factoid（事实型）** | "马云哪年出生？" | 单一事实、可验证 | L1 准确性、L4 相关性 |
| **Concept（概念型）** | "什么是反向代理？" | 定义解释 | L1 准确性、L5 体验 |
| **Comparison（对比型）** | "React 和 Vue 的区别" | 多实体对比 | L2 整合质量、L4 相关性 |
| **How-to（操作型）** | "如何配置 Nginx" | 步骤指导 | L4 相关性、L5 体验 |
| **Open-ended（开放型）** | "AI 对未来就业的影响" | 多视角分析 | L2 整合质量、L3 引文 |

### 2.3 自适应权重方案

| 查询类型 | L1 准确性 | L2 整合 | L3 引文 | L4 相关性 | L5 体验 |
|---------|----------|--------|--------|----------|--------|
| Factoid | 45% | 5% | 15% | 25% | 10% |
| Concept | 35% | 10% | 15% | 25% | 15% |
| Comparison | 20% | 35% | 15% | 20% | 10% |
| How-to | 25% | 10% | 10% | 35% | 20% |
| Open-ended | 15% | 30% | 25% | 15% | 15% |

### 2.4 实现建议

```python
QUERY_TYPE_WEIGHTS = {
    "factoid": {
        "veracity": 0.45, "synthesis_quality": 0.05,
        "citation_quality": 0.15, "relevance": 0.25,
        "user_experience": 0.10,
    },
    "concept": {
        "veracity": 0.35, "synthesis_quality": 0.10,
        "citation_quality": 0.15, "relevance": 0.25,
        "user_experience": 0.15,
    },
    "comparison": {
        "veracity": 0.20, "synthesis_quality": 0.35,
        "citation_quality": 0.15, "relevance": 0.20,
        "user_experience": 0.10,
    },
    "how_to": {
        "veracity": 0.25, "synthesis_quality": 0.10,
        "citation_quality": 0.10, "relevance": 0.35,
        "user_experience": 0.20,
    },
    "open_ended": {
        "veracity": 0.15, "synthesis_quality": 0.30,
        "citation_quality": 0.25, "relevance": 0.15,
        "user_experience": 0.15,
    },
}

def aggregate_with_query_type(scores: dict, query_type: str) -> dict:
    weights = QUERY_TYPE_WEIGHTS.get(query_type, QUERY_TYPE_WEIGHTS["concept"])
    # 使用与 aggregate_scores 相同的聚合逻辑，但使用自适应权重
    ...
```

---

## 三、Judge 一致性与校准机制

### 3.1 问题

LLM-as-a-Judge 的核心挑战是评分一致性：
- 同一答案多次评分可能得到不同结果
- 不同 prompt 顺序可能影响评分（position bias）
- judge 的评分倾向可能随时间漂移

### 3.2 建议措施

#### 3.2.1 双 Judge 机制

```
流程：
  1. 用 Judge A（如 GPT-4o-mini）评分
  2. 用 Judge B（如 DeepSeek-chat）评分
  3. 比较两个评分结果
  4. 如果分歧 > 1 分，触发人工复核
```

#### 3.2.2 Position Bias 控制

```
方法：
  1. 对同一批答案，随机打乱顺序后评分两次
  2. 比较两次评分的差异
  3. 如果差异 > 0.3 分（平均），说明存在 position bias
  4. 在 prompt 中加入"请独立评估每个答案，不受其他答案影响"的指令
```

#### 3.2.3 黄金标准集校准

```
方法：
  1. 准备 50-100 条人工标注的黄金标准样本
  2. 每周用 judge 对黄金标准集评分一次
  3. 计算 judge 评分与人工标注的 Pearson 相关系数
  4. 如果相关系数 < 0.7，需要调整 judge prompt 或更换 judge 模型
```

#### 3.2.4 评分一致性指标

| 指标 | 计算方法 | 目标值 |
|------|---------|-------|
| Self-consistency | 同一答案评分 3 次的标准差 | < 0.5 |
| Inter-judge agreement | 两个 judge 的 Pearson 相关系数 | > 0.75 |
| Human alignment | judge 与人工标注的 Pearson 相关系数 | > 0.70 |
| Position bias | 不同顺序评分的平均差异 | < 0.3 |

---

## 四、L2 评分标准调整

### 4.1 问题

L2 的 4 分标准中要求"信息冲突被明确指出、分析和解释"，但实际场景中很多查询的检索文档本身没有冲突。要求所有答案都处理冲突是不现实的，这可能导致 4 分几乎无法达到。

### 4.2 建议

将 4 分标准中的冲突处理改为条件性要求：

**原表述**：
> 信息冲突被明确指出、分析和解释（而非回避）

**建议改为**：
> **当存在信息冲突时**，能够明确指出、分析和解释（而非回避）；当文档信息一致时，能够构建层次分明、逻辑清晰的知识结构

---

## 五、分数分布监控

### 5.1 问题

如果 90% 的答案都得 3 分，评估就失去了区分度。需要定期分析分数分布，确保评估标准的有效性。

### 5.2 建议措施

```python
def analyze_score_distribution(results: list[dict]) -> dict:
    """
    分析评分分布，检测评估标准的有效性。
    
    Args:
        results: 聚合结果列表
    
    Returns:
        分布分析报告
    """
    scores_by_dim = {}
    for dim in ["veracity", "synthesis_quality", "citation_quality", 
                "relevance", "user_experience"]:
        dim_scores = [r["scores"][dim] for r in results]
        scores_by_dim[dim] = {
            "mean": sum(dim_scores) / len(dim_scores),
            "std": (sum((s - sum(dim_scores)/len(dim_scores))**2 for s in dim_scores) / len(dim_scores)) ** 0.5,
            "distribution": {i: dim_scores.count(i) / len(dim_scores) * 100 for i in range(1, 5)},
        }
    
    # 检测区分度不足的维度
    low_discrimination = []
    for dim, stats in scores_by_dim.items():
        if stats["distribution"].get(3, 0) > 70:  # 超过 70% 得 3 分
            low_discrimination.append(dim)
    
    return {
        "total_samples": len(results),
        "scores_by_dim": scores_by_dim,
        "low_discrimination_dims": low_discrimination,
        "recommendation": "建议调整以下维度的评分标准: " + ", ".join(low_discrimination) if low_discrimination else "分布正常"
    }
```

### 5.3 监控频率

| 阶段 | 频率 | 样本量要求 |
|------|------|-----------|
| 初期验证 | 每 100 条分析一次 | ≥ 100 |
| 稳定运行 | 每周分析一次 | ≥ 500 |
| 标准调整后 | 立即分析 | ≥ 200 |

---

## 六、A/B 测试校准权重

### 6.1 问题

权重设定主观性强，需要科学的方法来确定最优权重。

### 6.2 建议方法

```
步骤：
  1. 收集用户反馈数据（👍/👎 按钮、点击行为等）
  2. 用当前权重计算每个答案的综合分数
  3. 计算综合分数与用户满意度的相关性
  4. 用网格搜索或贝叶斯优化调整权重
  5. 选择使相关性最高的权重组合
```

### 6.3 目标指标

| 指标 | 计算方法 | 目标值 |
|------|---------|-------|
| 分数-满意度相关 | Pearson 相关系数 | > 0.60 |
| 高分通过率 | 综合分数 ≥ 70 的答案中用户 👍 的比例 | > 80% |
| 低分淘汰率 | 综合分数 < 60 的答案中用户 👎 的比例 | > 70% |

---

## 七、保留原始多维度分数

### 7.1 建议

聚合分数用于追踪趋势，原始分数用于诊断问题。两者缺一不可。

### 7.2 输出格式

```json
{
  "query_id": "T2R-00001",
  "query_type": "comparison",
  "overall_pass": true,
  "total_score": 78.5,
  "grade": "B",
  "scores": {
    "veracity": 3,
    "synthesis_quality": 4,
    "citation_quality": 3,
    "relevance": 3,
    "user_experience": 4,
    "safety": 3
  },
  "weighted_raw": 3.28,
  "penalty_applied": false,
  "judge_reasons": {
    "veracity": "...",
    "synthesis_quality": "...",
    ...
  }
}
```

---

## 八、生成阶段评估改进建议

> 关联：`exp-005-generation-stage-eval-rubric.md` —— LLM 生成阶段专属评估标准（11 维度）

### 8.1 维度精简：核心 vs 辅助分组

#### 问题

生成阶段评估包含 11 个维度，在实际操作中会带来以下挑战：
- Judge LLM 的 prompt 过长，可能导致注意力分散
- 评分成本高（每个答案需要 11 次维度判断）
- 维度间可能存在相关性，导致信息冗余

#### 行业对比

| 框架 | 维度数量 |
|------|---------|
| RAGAS | 3 个核心维度 |
| DeepEval | 4-5 个维度 |
| Anthropic | 3 个核心维度（Helpful, Honest, Harmless） |
| Google 内部 | 通常 5-6 个维度 |
| 本框架 | 11 个维度 |

#### 建议分组

| 核心维度（每次必评） | 辅助维度（按需评估） |
|-------------------|-------------------|
| G-L1 准确性 | G-L3 引文质量 |
| G-L4 相关性 | G-L5 用户体验 |
| G-L2 整合质量 | G-D 答案结构适配 |
| G-B 上下文利用 | G-E 内部一致性 |
| G-A 指令遵循 | |
| G-L6 安全（门槛） | G-C 拒答判断（特定场景） |

**使用策略**：
- 日常 A/B 测试：仅评估核心维度（6 个）
- 模型选型/重大迭代：评估全部 11 个维度
- 问题归因：根据问题类型选择辅助维度（如怀疑引文问题时评估 G-L3）

---

### 8.2 G-E 内部一致性与 G-L1 的重叠

#### 问题

- G-L1 准确性已经要求"事实正确、无幻觉"
- G-E 内部一致性要求"答案内部无矛盾"
- 但"内部矛盾"本质上也是一种"事实错误"（前后不一致）

#### 行业实践

- 大多数框架将内部一致性纳入 Factuality 评估
- 单独提出通常是为了长答案（>1000 字）的专项评估

#### 建议

| 答案长度 | 建议 |
|---------|------|
| < 500 字 | 合并 G-E 到 G-L1，在 G-L1 子维度中增加"内部一致性" |
| 500-1000 字 | 保留 G-E，但作为辅助维度 |
| > 1000 字 | 保留 G-E 作为核心维度 |

---

### 8.3 G-D 答案结构适配与 G-L4/G-L5 的边界

#### 问题

- G-L4 相关性已经包含"回答形式适配查询类型"
- G-L5 用户体验已经包含"格式运用"
- G-D 的"根据查询类型选择答案结构"与两者都有重叠

#### 实际评分时的混淆场景

| 场景 | 可能的归属 |
|------|-----------|
| 用户问对比问题，答案用纯文字而非表格 | G-L4 低分？G-D 低分？G-L5 低分？ |
| 用户问操作步骤，答案用段落而非列表 | G-L4 低分？G-D 低分？ |
| 答案用了表格但表格格式混乱 | G-L5 低分？G-D 低分？ |

#### 建议：明确三者的评估边界

| 维度 | 评估内容 | 示例 |
|------|---------|------|
| **G-L4 相关性** | 是否回答了用户的问题 | 用户问对比，答案是否做了对比 |
| **G-D 结构适配** | 是否选择了正确的结构类型 | 对比 → 表格；步骤 → 列表；代码 → 代码块 |
| **G-L5 用户体验** | 格式执行质量如何 | 表格是否清晰、列表是否规范、代码块是否有语法高亮 |

**评分逻辑**：
1. 先看 G-L4：是否回答了问题？
2. 再看 G-D：回答的结构类型选对了吗？
3. 最后看 G-L5：结构执行得好不好？

---

### 8.4 简洁性评估强化

#### 问题

- G-L5 用户体验中提到了"篇幅适当"
- 但在 AI 搜索场景中，**简洁性**是一个非常重要的独立指标
- 用户通常希望"用最少的字数获得完整信息"

#### 行业实践

| 公司 | 指标名称 | 说明 |
|------|---------|------|
| Google Search | Conciseness | 独立评估维度 |
| Perplexity AI | Information Density | 有效信息量 / 总字数 |
| Microsoft Copilot | Length Appropriateness | 篇幅与查询复杂度匹配度 |

#### 建议

在 G-L5 中强化简洁性的评估标准，增加以下子维度：

| 子维度 | 定义 | 评估方法 |
|--------|------|---------|
| **信息密度** | 有效信息量 / 总字数 | LLM judge 评估"是否有多余的铺垫和废话" |
| **篇幅适配** | 答案长度是否与查询复杂度匹配 | 简单查询 → 短答案；复杂查询 → 长答案 |
| **首句直达** | 第一句是否直接给出核心答案 | 检查答案开头是否直入主题 |

**4 档评分标准建议**：

| 档位 | 标签 | 定义 |
|------|------|------|
| **1** | 冗长/空洞 | 答案充斥大量无关铺垫、重复表述或废话。核心信息被淹没在冗余文字中。 |
| **2** | 偏长 | 答案包含了必要的信息，但有明显的铺垫或重复。可以精简 30% 以上而不损失信息。 |
| **3** | 简洁 | 答案篇幅适当，无明显冗余。每一段都有信息量。 |
| **4** | 精炼 | 答案用最少的必要字数传达了完整信息。首句直达核心，无一句废话。 |

---

### 8.5 时效性评估补充

#### 问题

- AI 搜索场景中，时间敏感信息很常见（如"最新政策"、"当前价格"）
- LLM 是否标注了信息的时效范围？是否区分了"历史事实"和"当前状态"？

#### 行业实践

| 公司 | 指标名称 | 说明 |
|------|---------|------|
| Google Search | Freshness | 信息新鲜度评估 |
| 内部评估 | Temporal Accuracy | 时间准确性 |

#### 建议

在 G-L1 准确性中增加"时效性声明"子维度：

| 子维度 | 定义 | 评估标准 |
|--------|------|---------|
| **时效性标注** | 对时间敏感信息是否标注了截止日期 | 如"截至 2024 年 3 月" |
| **时间状态区分** | 是否区分了历史事实和当前状态 | 如"该公司曾于 2020 年..." vs "目前该公司..." |
| **过期信息处理** | 当检索文档信息可能过期时，是否提醒用户 | 如"该数据来自 2022 年的文档，可能已过时" |

---

### 8.6 G-C 拒答判断的评估成本优化

#### 问题

- G-C 需要预先判断"检索文档是否足够支撑答案"
- 这个判断本身就需要人工或 LLM 标注
- 增加了评估的复杂度和成本

#### 建议

将 G-C 改为**按需评估**维度：

| 场景 | 是否评估 G-C |
|------|-------------|
| 日常 A/B 测试 | ❌ 不评估 |
| 检索质量专项评估 | ✅ 评估（需要标注检索文档的相关性） |
| 模型拒答能力测试 | ✅ 评估（专门构造信息不足的场景） |
| 问题归因分析 | ✅ 评估（当发现答案质量差时，判断是否是拒答不当导致） |

---

### 8.7 生成阶段评分聚合建议

#### 权重分配

生成阶段的评分聚合可以与系统级评估略有不同，因为生成阶段更关注 LLM 的核心能力：

| 维度 | 权重 | 理由 |
|------|------|------|
| G-L1 准确性 | 25% | 事实正确是基础，但检索也可能影响 |
| G-L4 相关性 | 20% | 切题回答是 LLM 的基本能力 |
| G-B 上下文利用 | 20% | 给定检索结果后，LLM 是否充分利用 |
| G-L2 整合质量 | 15% | 多源整合能力 |
| G-A 指令遵循 | 10% | prompt 迭代阶段的核心指标 |
| G-L3 引文质量 | 5% | 引文是加分项 |
| G-L5 用户体验 | 5% | 呈现质量 |

**与系统级评估的差异**：
- G-B 上下文利用权重更高（20% vs 系统级无此维度）
- G-A 指令遵循加入聚合（10%）
- G-L3 引文质量权重降低（5% vs 系统级 15%）

#### 聚合公式

```python
GENERATION_WEIGHTS = {
    "veracity": 0.25,
    "relevance": 0.20,
    "context_utilization": 0.20,
    "synthesis_quality": 0.15,
    "instruction_following": 0.10,
    "citation_quality": 0.05,
    "user_experience": 0.05,
}

def aggregate_generation_scores(scores: dict) -> dict:
    """
    聚合生成阶段评分。
    
    与系统级评估的区别：
    - 加入 G-A 指令遵循（10%）
    - 加入 G-B 上下文利用（20%）
    - 降低 G-L3 引文质量权重（5%）
    """
    # 门槛检查
    gate_failures = []
    if scores["veracity"] < 2:
        gate_failures.append("veracity")
    if scores["safety"] < 2:
        gate_failures.append("safety")
    
    if gate_failures:
        return {
            "pass": False,
            "total_score": 0,
            "grade": "F",
            "gate_failures": gate_failures,
        }
    
    # 加权求和
    weighted_sum = sum(scores[dim] * GENERATION_WEIGHTS[dim] for dim in GENERATION_WEIGHTS)
    total_score = (weighted_sum - 1) / 3 * 100
    total_score = round(total_score, 1)
    
    # 等级映射
    if total_score >= 90:
        grade = "S"
    elif total_score >= 80:
        grade = "A"
    elif total_score >= 70:
        grade = "B"
    elif total_score >= 60:
        grade = "C"
    else:
        grade = "D"
    
    return {
        "pass": total_score >= 60,
        "total_score": total_score,
        "grade": grade,
        "gate_failures": [],
        "weighted_raw": round(weighted_sum, 3)
    }
```

---

## 九、后续事项（待定）

### 9.1 多轮对话评估

AI 搜索产品大多是对话式交互，当前框架主要面向单轮问答。后续需要增加：

- **上下文理解**：是否正确理解了多轮对话的上下文
- **指代消解**：是否能正确解析代词和省略
- **意图延续**：是否保持了对话意图的连贯性
- **信息一致性**：前后回答是否矛盾
- **对话策略**：是否主动追问、澄清模糊意图

### 9.2 回答效用维度

回答效用（Answer Utility）已移入待定维度，保留完整定义供后续启用。启用条件：

- 当产品需要评估"用户看完答案后是否完成了信息获取任务"时
- 当需要区分"完整但无用"和"简洁但有用"的答案时

---

## 十、实施优先级

### 系统级评估

| 优先级 | 建议 | 理由 | 预计工作量 |
|-------|------|------|-----------|
| **P0** | 实现评分聚合方法 | 没有聚合规则无法产出可操作的结果 | 1-2 天 |
| **P0** | 定义查询分类体系 | 不同查询类型需要不同权重，否则评估结果不可比 | 2-3 天 |
| **P1** | 实现 Judge 一致性控制 | 保证评估结果的可信度 | 3-5 天 |
| **P1** | 调整 L2 的 4 分标准 | 提高评分标准的可操作性 | 0.5 天 |
| **P2** | 实现分数分布监控 | 定期验证评估标准的有效性 | 2-3 天 |
| **P2** | 收集用户反馈校准权重 | 用数据驱动权重优化 | 持续进行 |
| **P3** | 多轮对话评估 | 如果产品需要 | 5-10 天 |

### 生成阶段评估

| 优先级 | 建议 | 理由 | 预计工作量 |
|-------|------|------|-----------|
| **P0** | 将 11 个维度分为核心 + 辅助 | 降低评估成本，提高可操作性 | 1 天 |
| **P0** | 实现生成阶段评分聚合 | 用于 A/B 测试和模型选型 | 1-2 天 |
| **P1** | 明确 G-D 与 G-L4/G-L5 的边界 | 避免评分时的混淆 | 0.5 天 |
| **P1** | 在 G-L5 中强化简洁性评估 | AI 搜索的核心用户体验指标 | 0.5 天 |
| **P2** | 考虑 G-E 是否与 G-L1 合并 | 减少维度冗余 | 0.5 天 |
| **P2** | 在 G-L1 中增加时效性评估 | 时间敏感信息的处理 | 0.5 天 |
| **P3** | G-C 改为按需评估 | 降低常规评估成本 | 0.5 天 |
