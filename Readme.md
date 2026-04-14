# Industry AI Demo

## AAMI 良率智能助手 MVP Demo
主入口 Notebook：`notebooks/aami_yield_ai_copilot_demo.ipynb`

## 1) 安装依赖
建议在 Python 3.10+ 环境安装：

```bash
pip install pandas numpy matplotlib scikit-learn jupyter openai google-genai python-dotenv
```

## 2) 环境变量配置（可选）
### OpenAI
```bash
export OPENAI_API_KEY="your_openai_api_key"
# 可选
export OPENAI_MODEL="gpt-4.1-mini"
```

### Gemini（任选其一）
```bash
export GEMINI_API_KEY="your_gemini_api_key"
# 或
export GOOGLE_API_KEY="your_google_api_key"
# 可选
export GEMINI_MODEL="gemini-2.0-flash"
```

### Ollama（本地模型）
```bash
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen2.5:3b"   # 例如 qwen2.5:7b-instruct / gemma3:4b
```

本地启动示例：
```bash
ollama serve
ollama pull qwen2.5:3b
```

> 代码会自动读取环境变量；不会在源码或 Notebook 中硬编码密钥。

### 使用 `.env` 文件（推荐）
也可以在项目根目录创建 `.env`：

```bash
OPENAI_API_KEY=your_openai_api_key
# 或
GEMINI_API_KEY=your_gemini_api_key
# 或
GOOGLE_API_KEY=your_google_api_key
# 可选
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

`src/llm_provider.py` 会自动尝试 `load_dotenv()`，因此可以直接从 `.env` 读取 key（未安装 `python-dotenv` 时会自动跳过，不影响 local fallback）。

## 3) 启动演示
```bash
jupyter notebook notebooks/aami_yield_ai_copilot_demo.ipynb
```

## 4) LLM 解释开关与 provider 逻辑
Notebook 中提供：

```python
USE_LLM = True
LLM_PROVIDER = "auto"  # auto / openai / gemini / ollama / local
```

行为逻辑：
- `USE_LLM=False`：强制使用本地模板解释。
- `USE_LLM=True` 且 `LLM_PROVIDER="auto"`：
  1. 先尝试 OpenAI（存在 `OPENAI_API_KEY`）
  2. 否则尝试 Gemini（存在 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`）
  3. 否则尝试 Ollama（本地服务可达）
  4. 最后回退 local fallback

## 5) 在 Notebook 看实际调试信息
“风险解释”和“建议动作”输出中会打印以下字段：
- `provider`（最终实际 provider）
- `attempted_provider`（最初尝试 provider）
- `fallback_used`
- `fallback_reason`
- `error`

即使发生 fallback，也能看到失败原因，不再静默。

## 6) 数据文件
- 原始数据：`data/aami_yield_demo.csv`
- 清洗数据：`data/aami_yield_demo_clean.csv`

均可直接用于 Demo 演示。
