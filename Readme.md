# Industry AI Demo

## AAMI 良率智能助手 MVP Demo
主入口 Notebook：`notebooks/aami_yield_ai_copilot_demo.ipynb`

## 1) 安装依赖
建议在 Python 3.10+ 环境安装：

```bash
pip install pandas numpy matplotlib scikit-learn jupyter openai google-genai
```

## 2) 环境变量配置（可选）
### OpenAI
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### Gemini（任选其一）
```bash
export GEMINI_API_KEY="your_gemini_api_key"
# 或
export GOOGLE_API_KEY="your_google_api_key"
```

> 代码会自动读取环境变量；不会在源码或 Notebook 中硬编码密钥。

## 3) 启动演示
```bash
jupyter notebook notebooks/aami_yield_ai_copilot_demo.ipynb
```

## 4) LLM 解释开关
Notebook 中提供：

```python
USE_LLM = True
LLM_PROVIDER = "auto"  # auto / openai / gemini / local
```

行为逻辑：
- `USE_LLM=False`：强制使用本地模板解释。
- `USE_LLM=True` 且 `LLM_PROVIDER="auto"`：
  1. 先尝试 OpenAI（存在 `OPENAI_API_KEY`）
  2. 否则尝试 Gemini（存在 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`）
  3. 都没有则使用 local fallback

Notebook 会打印当前解释方式：
- 当前解释由 OpenAI 生成
- 当前解释由 Gemini 生成
- 当前解释由本地模板生成

## 5) 数据文件
- 原始数据：`data/aami_yield_demo.csv`
- 清洗数据：`data/aami_yield_demo_clean.csv`

均可直接用于 Demo 演示。
