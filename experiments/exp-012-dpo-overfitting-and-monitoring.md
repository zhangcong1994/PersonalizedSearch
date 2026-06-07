# DPO 过拟合风险与训练监控指南

> 来源：exp-012 DPO 训练讨论总结
>
> 适用场景：自对比式 DPO（self-generated + judge-scored pairs）
>
> 创建日期：2026-06-04

---

## 一、DPO 常见过拟合模式

### 1. Reward Hacking / 长度利用（Length Exploitation）

模型发现生成更长的回答就能获得更高的隐式 reward，response 越来越长、越来越啰嗦。

**检测方法**：
- 构造数据时检查 chosen/rejected 长度比，>1.5 即预警
- 训练时监控 KL 散度和 response 实际长度变化趋势

### 2. 偏好崩溃（Preference Collapse）

模型对所有 prompt 都输出同一套"万能回答"，过度拟合偏好信号。

**检测方法**：
- DPO accuracy 持续 > 0.98
- 验证集上生成多样性骤降

### 3. Chosen/Rejected 区分度爆炸

chosen 和 rejected 的 log-probability 差值持续拉大（>10），死记偏好信号。

**检测方法**：
- 监控 `rewards/margins`，持续 > 10 是过拟合信号
- 监控 `logps/chosen` 和 `logps/rejected` 的差值趋势

### 4. 对 Judge Model 过拟合

偏好对由 judge model 生成，模型学到的是"讨好 judge"的 pattern，而非真正的偏好。

**检测方法**：
- 换一个 judge model 交叉验证
- 在未见过的 query 上评估泛化性

---

## 二、过拟合缓解策略

### 2.1 数据层面

| 策略 | 说明 | 本实验建议 |
|------|------|-----------|
| 增加数据量和多样性 | 过拟合最常见的根源 | 目标 300-800 对（可行性验证），1K-5K 对（稳定提升） |
| 均衡长度分布 | 避免 chosen 系统性比 rejected 长 | 构造数据时打印 length ratio，>1.5 时筛选过滤 |
| 偏好对质量过滤 | 降低 judge 噪声 | `min_gap` 阈值 5-10，`min_chosen_score` 过滤低分 pair |
| 多策略混合 | best_vs_worst + best_vs_median 混合 | `--strategies best_vs_worst,best_vs_median` |

### 2.2 训练超参（小数据量适配）

| 超参 | 常规值 | 小数据量（<500对）推荐 | 本实验默认 |
|------|--------|----------------------|-----------|
| `beta` | 0.1 | 0.3~0.5 | 0.1 |
| `epochs` | 1~3 | **1** | 1 |
| `LoRA r` | 8~16 | 4~8 | 8 |
| `lr` | 5e-5 | 1e-5~2e-5 | 5e-5 |
| `warmup_ratio` | 0.1 | 0.15~0.2 | 0.1 |
| `lora_dropout` | 0.05~0.1 | 0.1~0.15 | 0.1 |

### 2.3 训练技巧

- **precompute_ref_log_probs=True**：已启用，预计算后释放 ref model 节省显存
- **gradient checkpointing**：已启用（`use_reentrant=False`，兼容 DPO）
- **验证集划分**：`--val-split 0.15`，周期性评估 `--eval-steps 30`
- **Cosine LR + Warmup**：已配置

---

## 三、训练监控体系

### 3.1 关键指标说明

| 指标 | 来源 | 正常趋势 | 过拟合信号 |
|------|------|----------|-----------|
| `loss` | DPO loss | 持续下降 → 持平 | 快速降到接近 0 |
| `rewards/chosen` | 隐式 reward | 小幅上升 | 暴涨 |
| `rewards/rejected` | 隐式 reward | 下降 | crash（骤降） |
| `rewards/margins` | chosen - rejected | 1~8 | **> 10 持续偏高** |
| `rewards/accuracies` | chosen>rejected 比例 | 0.65~0.95 | **> 0.98** |
| `kl` | 与 ref model 的 KL | 缓慢增长 | **> 5 或暴涨** |
| `logps/chosen` | chosen 的 log prob | 小幅上升或持平 | **持续下降（遗忘）** |
| `logps/rejected` | rejected 的 log prob | 下降 | 变化趋势异常 |

### 3.2 监控工具

| 工具 | 位置 | 内容 |
|------|------|------|
| 实时日志 WARNING | 训练 stdout | margin/accuracy/KL/logp 异常告警 |
| `dpo_metrics.json` | `{output_dir}/` | 每步完整指标序列，用于事后分析 |
| `data_stats.json` | `{output_dir}/` | 数据集统计（gap/gap/长度分布） |
| `eval_results.json` | `{output_dir}/` | 验证集最终/周期性评估结果 |
| `dpo_pairs_report.json` | `data/processed/exp012/` | 构造数据时的质量报告（by `construct_dpo_pairs.py`） |

### 3.3 告警规则（`DPOMonitorCallback`）

| 告警条件 | 级别 | 建议动作 |
|----------|------|---------|
| `rewards/margins > 10` 连续 2 步 | WARNING | 增大 beta 或减少训练 |
| `rewards/accuracies > 0.98` | WARNING | 可能死记硬背，检查验证集表现 |
| `KL > 5` | WARNING | 偏离 ref model 过大，增大 beta |
| `logps/chosen` 连续下降 4+ 步 | WARNING | 模型遗忘好行为，检查数据质量 |
| `chosen/rejected len ratio > 1.5`（数据加载时）| WARNING | 有长度 exploitation 风险，训练时考虑长度惩罚 |

### 3.4 训练后分析

训练完成后 `DPOMonitorCallback` 会自动打印：

```
============================================================
  DPO 训练指标终态
============================================================
  rewards/margins         :    X.XXX →    Y.YYY  (ΔZ.ZZZ)
  rewards/accuracies      :    X.XXX →    Y.YYY  (ΔZ.ZZZ)
  kl                      :    X.XXX →    Y.YYY  (ΔZ.ZZZ)
  logps/chosen            :    X.XXX →    Y.YYY  (ΔZ.ZZZ)
  logps/rejected          :    X.XXX →    Y.YYY  (ΔZ.ZZZ)
  loss                    :    X.XXX →    Y.YYY  (ΔZ.ZZZ)
  ✓/⚠️  Margin trend: +Z.Z — 合理范围/过拟合风险/训练崩溃
============================================================
```

---

## 四、数据构造阶段预警

`construct_dpo_pairs.py` 在生成数据时会自动检查：

| 检查项 | 阈值 | 含义 |
|--------|------|------|
| 长度比 > 1.5 | chosen 系统性比 rejected 长 | DPO 可能学到长度 exploitation |
| 长度比 < 0.7 | rejected 比 chosen 长 | 偏好信号可能反向 |
| 平均 gap > 30 | 偏好信号过强 | DPO 易过拟合 |
| 平均 gap < 5 | 偏好信号太弱 | DPO 可能学不到东西 |

### 理想的数据质量范围

```
gap:       5 ~ 30（chosen 与 rejected 的 judge 分差）
len_ratio: 0.8 ~ 1.3（chosen 与 rejected 的长度比）
```

---

## 五、数据量参考

| 场景 | 4B 模型 | 8B 模型 | 说明 |
|------|---------|---------|------|
| 最小可跑通（smoke test） | 50~100 对 | 50~100 对 | 能看到 loss 下降，无泛化价值 |
| 快速验证可行性 | 300~800 对 | 500~1500 对 | 能观察到明显效果提升 |
| 稳定提升 | 1K~5K 对 | 2K~10K 对 | 泛化好、不易过拟合 |
| 高多样性覆盖 | 5K~20K+ | 10K~50K+ | 多 domain 覆盖 |

---

## 六、自对比式 DPO 特别注意

当前实验使用「同一模型自己生成多个回答 + Judge 打分 → 选 best/worst 构造 pair」的方式（self-generated + judge-scored）。

**额外风险**：
1. **确认偏误（Confirmation Bias）**：模型自评可能放大已有偏好
2. **分布漂移**：每轮训练后模型变了，旧数据"过时"
3. **多样性不足**：低温度采样下 chosen/rejected 差异太小

**缓解措施**：
- 使用独立的强 Judge（已用 deepseek-chat）
- 高温采样（T=0.8）保证生成多样性
- 每轮迭代重新采样（iterative/online DPO）
- 对 judge score 差值设阈值，只保留差距足够大的 pair

---

## 七、训练的检查清单

- [ ] 数据：chosen/rejected 长度比在 0.8~1.3 之间
- [ ] 数据：平均 gap 在 5~30 之间
- [ ] 数据：数据量 ≥ 300 对（可行性验证）
- [ ] 训练：小数据量时 beta ≥ 0.3
- [ ] 训练：epochs = 1（小数据量）
- [ ] 训练：lr ≤ 2e-5（小数据量）
- [ ] 训练：划分验证集（--val-split 0.15）
- [ ] 监控：`rewards/margins` 稳定在 1~8
- [ ] 监控：`kl` 稳定在 0~3
- [ ] 监控：`rewards/accuracies` 在 0.65~0.95
- [ ] 监控：`logps/chosen` 不持续下降
- [ ] 评估：换未见过的 query 做泛化测试
