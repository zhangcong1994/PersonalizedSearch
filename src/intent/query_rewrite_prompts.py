"""
Query rewrite prompt registry for exp-002 experiments.

Each entry defines:
  - strategy: "none" | "single" | "multi_query" | "hyde" | "hyde_rrf" | "prf"
  - system: system prompt template
  - human: human message template ({query} is the placeholder)
  - max_tokens: max output tokens for LLM
  - output_parser: "text" | "json_list" | "json_obj"
  - rrf_k: RRF k parameter (multi_query / hyde_rrf strategies only)
  - hyde_answer_len: target answer length in characters (hyde strategies only)

Usage:
    from src.intent.query_rewrite_prompts import REGISTRY, get_experiment_config
    cfg = get_experiment_config("E2a-B1")
"""

# ── E2a: Prompt iteration prompts ────────────────────────

_E2A_B1_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请将其扩展为更完整、更具体的搜索词，以提高信息检索的召回率。

规则：
1. 如果查询本身已经足够清晰完整（>15字），直接返回原查询
2. 如果查询是缩写、口语化、省略了关键信息，补全为完整的疑问句或陈述句
3. 保持改写后的查询为一行纯文本，不要加引号、编号、解释
4. 不要编造查询中没有的信息

示例：
原始: 蜂巢取快递验证码摁错怎么办
改写: 蜂巢快递柜取件验证码输入错误如何重新获取

原始: 生产过后怎么还有一层肚子
改写: 产后腹部脂肪堆积原因及恢复平坦小腹的方法

原始: 考研英语一和英语二有什么区别
改写: 考研英语一和英语二的区别 考试内容 难度对比

原始: 怎么判断鱼卵是否活着
改写: 如何判断鱼卵是否存活 鱼卵活性检测方法

原始: 比特币和以太坊哪个更值得投资
改写: 比特币和以太坊哪个更值得投资

原始: 西红柿炒鸡蛋的正确做法是什么
改写: 西红柿炒鸡蛋的正确做法步骤

原始: 为什么晚上睡觉会磨牙
改写: 晚上睡觉磨牙的原因 夜磨牙症病因"""

_E2A_P1_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请将其扩展为更完整、更具体的搜索词，以提高信息检索的召回率。

规则：
1. 如果查询本身已经足够清晰完整（>15字），直接返回原查询；如果 < 15 字，改写为 20-40 字的完整表达
2. 如果查询包含缩写，必须展开为全称
3. 如果查询是口语化表达，必须补充正式术语作为同义词
4. 至少添加 2 个相关概念词
5. 保留原查询中的所有核心名词，不要删除或替换
6. 保持改写后的查询为一行纯文本，不要加引号、编号、解释
7. 不要编造查询中没有的信息

示例：
原始: 蜂巢取快递验证码摁错怎么办
改写: 蜂巢快递柜取件验证码输入错误如何重新获取验证码

原始: 生产过后怎么还有一层肚子
改写: 产后腹部脂肪堆积原因及恢复平坦小腹的方法

原始: 考研英语一和英语二有什么区别
改写: 考研英语一和英语二的区别 考试内容 难度对比

原始: 怎么判断鱼卵是否活着
改写: 如何判断鱼卵是否存活 鱼卵活性检测方法

原始: 比特币和以太坊哪个更值得投资
改写: 比特币和以太坊哪个更值得投资

原始: 西红柿炒鸡蛋的正确做法是什么
改写: 西红柿炒鸡蛋的正确做法步骤

原始: 为什么晚上睡觉会磨牙
改写: 晚上睡觉磨牙的原因 夜磨牙症病因"""

_E2A_P2_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请将其扩展为更完整、更具体的搜索词，以提高信息检索的召回率。

规则：
1. 如果查询是缩写、口语化、省略了关键信息，补全为完整的疑问句或陈述句
2. 保持改写后的查询为一行纯文本，不要加引号、编号、解释
3. 不要编造查询中没有的信息

示例：
原始: 为什么脸上老长痘痘
改写: 面部反复长痘的原因 痤疮成因 皮肤护理方法

原始: 怎么看电脑配置
改写: 如何查看电脑硬件配置 查看CPU型号内存大小显卡型号方法

原始: 苹果和安卓哪个好用
改写: 苹果iOS和安卓Android系统优缺点对比 适用人群

原始: 什么是区块链
改写: 区块链技术定义 分布式账本原理 去中心化特点

原始: 中国未来经济发展趋势
改写: 中国经济发展趋势 未来增长动力 产业结构转型

原始: 武汉有什么好玩的
改写: 武汉旅游景点推荐 武汉必去好玩的地方

原始: 怎么减肥最快
改写: 快速减肥方法 科学减重饮食运动计划

原始: 5G和4G有什么区别
改写: 5G和4G的区别 网速延迟应用场景对比"""

_E2A_P3_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请将其扩展为更完整、更具体的搜索词，以提高信息检索的召回率。

规则：
1. 如果查询本身已经足够清晰完整（>15字），直接返回原查询
2. 如果查询是缩写、口语化、省略了关键信息，补全为完整的疑问句或陈述句
3. 保持改写后的查询为一行纯文本，不要加引号、编号、解释
4. 不要编造查询中没有的信息

示例（每个示例展示好的改写和坏的改写）：
原始: 蜂巢验证码摁错
✗ 坏: 蜂巢快递柜验证码摁错    （太口语，缺乏术语，没有扩展信息）
✓ 好: 蜂巢快递柜取件验证码输入错误如何重新获取验证码

原始: 生产过后肚子
✗ 坏: 生产过后肚子怎么办        （仍然模糊，没有补充专业术语）
✓ 好: 产后腹部脂肪堆积原因及恢复平坦小腹的方法

原始: 苹果安卓哪个好
✗ 坏: 苹果和安卓哪个更好用      （仅重复原意，没有扩展对比维度）
✓ 好: 苹果iOS和安卓Android系统优缺点对比 适用人群分析

原始: 减肥最快
✗ 坏: 怎么减肥最快              （口语化，没有补充科学方法）
✓ 好: 快速减肥方法 科学减重饮食运动计划

原始: 5G 4G 区别
✗ 坏: 5G和4G有什么区别          （仅展开句式，无实质补充）
✓ 好: 5G和4G网络的区别 网速延迟应用场景对比

原始: 武汉好玩
✗ 坏: 武汉有什么好玩的地方      （仅补全句式）
✓ 好: 武汉旅游景点推荐 武汉必去好玩的地方攻略

原始: 区块链是什么
✗ 坏: 什么是区块链技术          （仅补全句式）
✓ 好: 区块链技术定义 分布式账本原理 去中心化特点 应用场景

原始: 脸上长痘
✗ 坏: 为什么脸上会长痘痘        （仅补全句式）
✓ 好: 面部反复长痘的原因 痤疮成因 皮肤护理方法"""

_E2A_P4_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请先判断查询类型，再根据类型采用不同的扩展策略。

查询类型与对应策略：
- 事实型查询（询问具体事实/数据/时间）：改写为完整疑问句，补充相关的时间、地点、人物等上下文信息
- 对比型查询（比较两个或多个事物）：明确对比双方的全称，添加对比维度关键词如"区别""优缺点""适用场景"
- 概念解释型查询（询问定义/概念/原理）：添加"定义""原理""特点""应用"等学术搜索术语
- 开放讨论型查询（趋势/影响/前景等）：生成包含多角度的搜索关键词串

规则：
1. 如果查询本身已经足够清晰完整（>15字），直接返回原查询
2. 保持改写后的查询为一行纯文本，不要加引号、编号、解释
3. 不要编造查询中没有的信息

示例：
原始: XX是哪一年成立的
改写(事实型): XX成立时间 创办年份 历史背景

原始: XX和YY哪个更好
改写(对比型): XX与YY的优缺点对比 适用场景分析 选择建议

原始: 什么是机器学习
改写(概念型): 机器学习定义 基本原理 分类方法 应用领域

原始: 人工智能未来发展趋势
改写(开放讨论): 人工智能未来发展趋势 技术突破方向 行业影响"""

_E2A_P5_SYSTEM = """你是一个搜索关键词提取器。用户输入的是真实搜索引擎中的短查询，请提取并扩展为关键词串用于信息检索。

规则：
1. 提取 3-7 个关键词或短短语，用空格分隔
2. 包含原查询中的所有核心词
3. 补充相关的扩展词（同义词、上位词、相关概念）
4. 优先使用名词和名词短语，避免虚词
5. 按重要性排序，核心词在前
6. 只输出关键词串，不要加引号、编号、解释

示例：
原始: 蜂巢取快递验证码摁错怎么办
输出: 蜂巢快递柜 取件验证码 输入错误 重新获取 操作方法

原始: 生产过后怎么还有一层肚子
输出: 产后恢复 腹部脂肪堆积 原因 瘦肚子 方法

原始: 考研英语一和英语二有什么区别
输出: 考研 英语一 英语二 区别 考试内容 难度 适用专业

原始: 怎么判断鱼卵是否活着
输出: 鱼卵 存活 判断方法 活性检测 鱼卵孵化

原始: 比特币和以太坊哪个更值得投资
输出: 比特币 以太坊 投资价值 对比 加密货币 风险

原始: 西红柿炒鸡蛋的正确做法
输出: 西红柿炒鸡蛋 做法 步骤 家常菜 烹饪技巧

原始: 为什么晚上睡觉会磨牙
输出: 夜磨牙症 原因 睡眠障碍 治疗方法"""


# ── E2b: Multi-Query prompts ─────────────────────────────

_E2B_M1_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请从以下两个角度生成改写版本，以提高信息检索的召回率。

改写角度：
1. 术语规范化：将查询中的口语化表达替换为正式术语，展开缩写
2. 上下文补全：补充查询隐含的时间、地点、场景等背景信息

输出格式（JSON）：
{{"sub_queries": ["术语规范化版本", "上下文补全版本"]}}

规则：
- 只输出 JSON，不要加任何其他文字
- 每条改写为一行纯文本，20-50 字
- 保留原查询的所有核心信息，不要编造

示例：
原始: 蜂巢验证码摁错
{{"sub_queries": ["蜂巢快递柜取件验证码输入错误", "蜂巢快递柜取件验证码输入错误如何重新获取"]}}

原始: 生产过后肚子
{{"sub_queries": ["产后腹部恢复 脂肪堆积", "产后腹部脂肪堆积原因及恢复平坦小腹的方法"]}}"""

_E2B_M2_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请从以下角度生成改写版本，以提高信息检索的召回率。

改写角度：
1. 术语规范化：将查询中的口语化表达替换为正式术语，展开缩写
2. 上下文补全：补充查询隐含的时间、地点、场景等背景信息
3. 抽象提问 (Step-back)：将具体查询抽象化为更高层的上层问题，上移一个概念层级

输出格式（JSON）：
{{"sub_queries": ["术语规范化版本", "上下文补全版本", "抽象上层问题"]}}

规则：
- 只输出 JSON，不要加任何其他文字
- 每条改写为一行纯文本，20-50 字
- 抽象问题应比原查询宽泛一个概念层级，能召回相关背景资料

示例：
原始: XX是哪一年成立的
{{"sub_queries": ["XX成立时间 创办年份", "XX公司的成立时间和创办背景", "XX的历史背景和创立过程"]}}

原始: 蜂巢验证码摁错
{{"sub_queries": ["蜂巢快递柜取件验证码输入错误", "蜂巢快递柜取件验证码输入错误如何重新获取", "快递柜取件操作流程和常见问题解决方法"]}}"""

_E2B_M3_SYSTEM = """你是一个搜索查询优化器。请先判断用户查询的类型，再按类型采用对应策略生成多条子查询。

【查询类型与策略】
- 事实型 (factual)：术语规范化 + 上下文补全 + Step-back 抽象问题
- 对比型 (comparison)：子查询分解（拆为基础介绍 + 对比维度 + 趋势展望）
- 概念解释型 (concept)：术语规范化 + 上下文补全 + 同义词扩展
- 开放讨论型 (open)：子查询分解（拆为多个维度的独立查询）

【输出格式】JSON：
事实型示例: {{"query_type": "factual", "sub_queries": ["术语版", "上下文版", "抽象问题"]}}
对比型示例: {{"query_type": "comparison", "sub_queries": ["XX介绍", "YY介绍", "XX YY区别", "XX趋势", "YY趋势"]}}
概念型示例: {{"query_type": "concept", "sub_queries": ["术语版", "上下文版", "同义词扩展版"]}}
开放型示例: {{"query_type": "open", "sub_queries": ["维度1", "维度2", "维度3", "维度4"]}}

规则：
- 只输出 JSON，不要加任何其他文字
- 每条子查询 15-50 字
- 对比型拆为 3-5 条子查询，开放型拆为 2-4 条子查询"""


# ── E2c: HyDE prompts ────────────────────────────────────

_E2C_H1_SYSTEM = """你是一个知识渊博的助手。请根据用户的问题，生成一段 100-150 字的回答。不需要保证完全准确，但应包含与该问题相关的关键概念、术语和背景信息。

要求：
1. 使用正式、信息丰富的语言
2. 包含与问题相关的专业术语和关键概念
3. 回答长度为 100-150 字
4. 就像你在写一个百科全书条目

用户问题: {query}
回答:"""

_E2C_H2_SYSTEM = _E2C_H1_SYSTEM

_E2C_H3_50 = """你是一个知识渊博的助手。请根据用户的问题，生成一段约 50 字的简短回答。不需要保证完全准确，但应包含与该问题相关的关键概念。

要求：
1. 使用正式、信息丰富的语言
2. 包含与问题相关的专业术语
3. 回答长度约 50 字
4. 就像你在写一个百科全书条目摘要

用户问题: {query}
回答:"""

_E2C_H3_200 = """你是一个知识渊博的助手。请根据用户的问题，生成一段约 200 字的详细回答。不需要保证完全准确，但应包含与该问题相关的关键概念、术语、背景信息和多角度分析。

要求：
1. 使用正式、信息丰富的语言
2. 包含与问题相关的专业术语和关键概念
3. 回答长度约 200 字
4. 就像你在写一个详细的百科全书条目

用户问题: {query}
回答:"""


# ── Experiment Registry ───────────────────────────────────

REGISTRY = {
    # ── E2a: Prompt iteration ──
    "E2a-B0": {
        "name": "no rewrite (baseline)",
        "strategy": "none",
        "system": None,
        "human": None,
        "max_tokens": 0,
        "output_parser": "text",
    },
    "E2a-B1": {
        "name": "v1 rewrite (current unified prompt)",
        "strategy": "single",
        "system": _E2A_B1_SYSTEM,
        "human": "原始: {query}\n改写:",
        "max_tokens": 128,
        "output_parser": "text",
    },
    "E2a-P1": {
        "name": "rich rules",
        "strategy": "single",
        "system": _E2A_P1_SYSTEM,
        "human": "原始: {query}\n改写:",
        "max_tokens": 128,
        "output_parser": "text",
    },
    "E2a-P2": {
        "name": "domain few-shot (T2Ranking examples)",
        "strategy": "single",
        "system": _E2A_P2_SYSTEM,
        "human": "原始: {query}\n改写:",
        "max_tokens": 128,
        "output_parser": "text",
    },
    "E2a-P3": {
        "name": "contrastive few-shot",
        "strategy": "single",
        "system": _E2A_P3_SYSTEM,
        "human": "原始: {query}\n改写:",
        "max_tokens": 128,
        "output_parser": "text",
    },
    "E2a-P4": {
        "name": "query-type-aware rewrite",
        "strategy": "single",
        "system": _E2A_P4_SYSTEM,
        "human": "原始: {query}\n改写:",
        "max_tokens": 128,
        "output_parser": "text",
    },
    "E2a-P5": {
        "name": "keyword expansion",
        "strategy": "single",
        "system": _E2A_P5_SYSTEM,
        "human": "原始: {query}\n输出:",
        "max_tokens": 128,
        "output_parser": "text",
    },

    # ── E2b: Multi-Query fusion ──
    "E2b-M1": {
        "name": "multi-angle rewrite (failure mode based)",
        "strategy": "multi_query",
        "system": _E2B_M1_SYSTEM,
        "human": "原始: {query}",
        "max_tokens": 256,
        "output_parser": "json_list",
        "rrf_k": 60,
    },
    "E2b-M2": {
        "name": "M1 + step-back (all types)",
        "strategy": "multi_query",
        "system": _E2B_M2_SYSTEM,
        "human": "原始: {query}",
        "max_tokens": 256,
        "output_parser": "json_list",
        "rrf_k": 60,
    },
    "E2b-M3": {
        "name": "type-differentiated strategy",
        "strategy": "multi_query",
        "system": _E2B_M3_SYSTEM,
        "human": "原始: {query}",
        "max_tokens": 384,
        "output_parser": "json_obj",
        "rrf_k": 60,
    },

    # ── E2c: HyDE ──
    "E2c-H1": {
        "name": "HyDE only",
        "strategy": "hyde",
        "system": _E2C_H1_SYSTEM,
        "human": None,
        "max_tokens": 300,
        "output_parser": "text",
        "hyde_answer_len": 150,
    },
    "E2c-H2": {
        "name": "HyDE + original query (RRF)",
        "strategy": "hyde_rrf",
        "system": _E2C_H2_SYSTEM,
        "human": None,
        "max_tokens": 300,
        "output_parser": "text",
        "hyde_answer_len": 150,
        "rrf_k": 60,
    },
    "E2c-H3": {
        "name": "HyDE answer length ablation",
        "strategy": "hyde_rrf",
        "system": _E2C_H1_SYSTEM,
        "human": None,
        "max_tokens": 300,
        "output_parser": "text",
        "hyde_answer_len": 100,
        "rrf_k": 60,
        "length_variants": {
            "50": _E2C_H3_50,
            "100": _E2C_H1_SYSTEM,
            "200": _E2C_H3_200,
        },
    },

    # ── E2d: PRF ──
    "E2d-P0": {
        "name": "no rewrite (baseline)",
        "strategy": "none",
        "system": None,
        "human": None,
        "max_tokens": 0,
        "output_parser": "text",
    },
    "E2d-P1": {
        "name": "PRF (TF-IDF, 5 terms)",
        "strategy": "prf",
        "system": None,
        "human": None,
        "max_tokens": 0,
        "output_parser": "text",
        "prf_top_k": 20,
        "prf_num_terms": 5,
        "prf_weighted": False,
    },
    "E2d-P2": {
        "name": "PRF (TF-IDF, 10 terms, weighted)",
        "strategy": "prf",
        "system": None,
        "human": None,
        "max_tokens": 0,
        "output_parser": "text",
        "prf_top_k": 20,
        "prf_num_terms": 10,
        "prf_weighted": True,
    },
}


def get_experiment_config(experiment_id: str) -> dict:
    if experiment_id not in REGISTRY:
        raise KeyError(
            f"Unknown experiment_id: {experiment_id}. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[experiment_id]


def list_experiments():
    return sorted(REGISTRY.keys())


def get_experiments_by_strategy(strategy: str) -> list[str]:
    return [eid for eid, cfg in REGISTRY.items() if cfg.get("strategy") == strategy]
