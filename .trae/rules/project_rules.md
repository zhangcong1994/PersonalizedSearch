# 项目规则

## 环境
- Python 3.11
- 虚拟环境: `.venv`

## 运行命令

```powershell
# 激活虚拟环境 (PowerShell)
.\.venv\Scripts\Activate.ps1

# 激活虚拟环境 (CMD)
.\.venv\Scripts\activate.bat

# 安装依赖
pip install -r requirements.txt

# 运行脚本
python scripts/<script_name>.py

# 启动 Gradio 应用
python app.py
```

## 项目结构
- `scripts/` - 数据处理和实验脚本
- `src/` - 核心库代码
- `experiments/` - 实验配置和分析文档
- `models/` - 本地模型缓存
- `data/` - 数据文件
- `config.yaml` - 项目配置（模型注册、索引、检索参数等）
- `progress.yaml` - 实验进度记录

## 代码规范
- 函数使用 snake_case 命名
- 所有公开函数必须有类型注解
- 日志使用 loguru
- 配置通过 `config.yaml` 和 `src/utils/config.py` 读取
- **数据盘规则**：所有涉及文件落盘的操作（下载数据、下载模型、保存训练后模型、生成训练数据、保存实验结果、写入向量数据库等），必须从 `src.utils.config` 导入 `DATA_ROOT`，并以 `DATA_ROOT` 为根目录构建输出路径。`DATA_ROOT` 通过环境变量 `PERSONALIZEDSEARCH_DATA_ROOT` 配置，未设置时回退到项目根目录。禁止将模型、数据、实验结果写入项目代码目录。
