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
- 开发机与服务器的数据盘路径不同，代码需要配置数据路径。所有数据、模型、实验结果都放到数据盘中。
