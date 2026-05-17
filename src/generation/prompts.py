import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


SYSTEM_PROMPTS = {
    "default": (
        "你是一个基于知识库回答问题的助手。请根据以下检索到的维基百科资料来回答问题。\n"
        "如果资料不足以回答问题，请如实说明，不要编造信息。\n"
        "回答时请注明引用的资料来源（文档标题）。"
    ),
    "factual": (
        "你是一个精确的知识问答引擎。请根据以下检索资料，给出简洁准确的事实性回答。\n"
        "要求：\n"
        "1. 先给出直接答案（一句话）\n"
        "2. 再提供 2-3 句补充说明\n"
        "3. 在相关句子末尾用 [来源: 文档标题] 注明出处\n"
        "4. 如果资料中没有确切答案，直接说「未找到相关信息」，不要猜测"
    ),
    "concept": (
        "你是一个概念解释专家。请根据以下检索资料，对一个概念进行清晰的定义和解释。\n"
        "要求：\n"
        "1. 先给出核心定义（1-2 句）\n"
        "2. 再展开关键要点（用 bullet points）\n"
        "3. 每个要点后标注 [来源: 文档标题]\n"
        "4. 如果信息不足，注明哪些内容是资料覆盖的，哪些是推理补充的"
    ),
    "open_discussion": (
        "你是一个知识综合与评述专家。请根据以下检索资料，对一个开放性问题给出综合性的分析和评述。\n"
        "要求：\n"
        "1. 先给出总体概述（2-3 句）\n"
        "2. 然后分 2-4 个角度展开讨论，每个角度冠以小标题\n"
        "3. 每个角度中引用相关资料并注明 [来源: 文档标题]\n"
        "4. 最后给出简要总结或展望"
    ),
    "comparison": (
        "你是一个对比分析专家。请根据以下检索资料，对两个或多个事物进行系统地对比分析。\n"
        "要求：\n"
        "1. 先简要介绍各方的基本情况（各1句）\n"
        "2. 用表格形式列出核心对比维度（维度 | 甲方 | 乙方）\n"
        "3. 表格中每个数据点注明 [来源: 文档标题]\n"
        "4. 最后总结各方适用场景或优劣"
    ),
}


CONTEXT_TEMPLATE = (
    "参考资料:\n"
    "{context}"
)

QUESTION_TEMPLATE = (
    "用户问题: {question}\n\n"
    "请根据以上参考资料回答问题："
)


FEW_SHOT_EXAMPLES = {
    "factual": """
示例问答：
参考资料:
[1] 来源: 阿尔伯特·爱因斯坦
阿尔伯特·爱因斯坦（1879年3月14日—1955年4月18日），德国出生的理论物理学家...

用户问题: 爱因斯坦什么时候出生？

回答:
爱因斯坦出生于1879年3月14日 [来源: 阿尔伯特·爱因斯坦]。他是公认的二十世纪最伟大的物理学家之一，提出了狭义相对论和广义相对论。

---
""",
    "concept": """
示例问答：
参考资料:
[1] 来源: 光合作用
光合作用是植物、藻类和某些细菌利用叶绿素将光能转化为化学能的过程...

用户问题: 什么是光合作用？

回答:
光合作用是植物等生物利用光能将二氧化碳和水转化为有机物并释放氧气的过程 [来源: 光合作用]。

关键要点：
- 需要光能和叶绿素参与 [来源: 光合作用]
- 反应方程式：6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂ [来源: 光合作用]
- 是地球上最重要的化学反应之一，维持大气氧含量

---
""",
}

SYSTEM_PROMPTS_WITH_SHOT = {
    key: prompt + "\n\n" + FEW_SHOT_EXAMPLES.get(key, "")
    for key, prompt in SYSTEM_PROMPTS.items()
}


USER_PROMPT_TEMPLATES = {
    "with_context": CONTEXT_TEMPLATE + "\n\n" + QUESTION_TEMPLATE,
    "bare": QUESTION_TEMPLATE,
}


class PromptManager:
    def __init__(self):
        self.system_prompts = dict(SYSTEM_PROMPTS_WITH_SHOT)
        self.user_template = USER_PROMPT_TEMPLATES["with_context"]

    def get_system_prompt(self, query_type: str = "default") -> str:
        return self.system_prompts.get(query_type, self.system_prompts["default"])

    def format_context(self, docs: List) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            title = doc.metadata.get("source", "未知")
            content = doc.page_content[:600]
            formatted.append(f"[{i+1}] 来源: {title}\n{content}")
        return "\n\n".join(formatted)

    def build_user_prompt(self, question: str, context: str) -> str:
        return self.user_template.format(context=context, question=question)

    def classify_query_type(self, question: str) -> str:
        keywords = {
            "factual": ["什么时候", "多少", "谁", "哪里", "哪一年", "日期", "年龄", "多高", "多长"],
            "comparison": ["区别", "对比", "不同", "差异", " vs ", "比较", "哪个更好", "优缺点"],
            "concept": ["什么是", "定义", "概念", "解释", "什么意思", "原理"],
            "open_discussion": ["发展趋势", "影响", "未来", "前景", "历程", "历史", "发展"],
        }
        for qtype, kws in keywords.items():
            if any(kw in question for kw in kws):
                return qtype
        if question.endswith("？") or question.endswith("?"):
            return "open_discussion"
        return "default"

    def add_template(self, name: str, system_prompt: str, few_shot: str = ""):
        self.system_prompts[name] = system_prompt + ("\n\n" + few_shot if few_shot else "")

    def remove_template(self, name: str):
        if name in self.system_prompts and name != "default":
            del self.system_prompts[name]


_default_manager = None


def get_default_prompts() -> PromptManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptManager()
    return _default_manager
