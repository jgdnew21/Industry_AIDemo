"""LLM provider wrapper for AAMI yield demo.

设计目标：
1) 在 notebook demo 场景下保持「稳定可跑」：无论云端 API 是否可用，都能回退到 local 模板。
2) 提升「可排错性」：显式返回 provider 选择、回退原因、原始错误、raw_text 等调试信息。
3) 统一 provider 链路：auto 模式优先顺序为 OpenAI -> Gemini -> Ollama -> local。
"""

from __future__ import annotations

# 标准库导入：
# - importlib.util: 用于“软依赖检测”（例如是否安装 openai/google-genai/dotenv）。
# - urllib: 使用标准库 HTTP 调用 Ollama，减少额外依赖。
import importlib.util
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

if importlib.util.find_spec("requests") is not None:
    import requests
else:
    requests = None


# ------------------------------
# 默认模型与默认地址（可被环境变量覆盖）
# ------------------------------
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "gemma4:e4b"
OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


# ------------------------------
# 环境变量加载
# ------------------------------
def _load_env_from_dotenv() -> None:
    """Best-effort 读取 .env。

    注意：
    - 若 python-dotenv 未安装，直接跳过，不影响 demo。
    - 任何异常都吞掉，避免 demo 因环境配置小问题崩掉。
    """
    if importlib.util.find_spec("dotenv") is None:
        return

    try:
        from dotenv import load_dotenv

        # override=False：尊重系统环境变量优先级。
        load_dotenv(override=False)
    except Exception:
        # 保持演示稳定，不把 dotenv 异常扩散到主流程。
        return


# 模块导入时自动尝试加载 .env。
_load_env_from_dotenv()


# ------------------------------
# 小工具函数（文本裁剪、key 检测）
# ------------------------------
def _trim_raw_text(text: Any, max_len: int = 1200) -> str:
    """截断原始文本，避免 notebook 打印过长内容影响可读性。"""
    if text is None:
        return ""
    raw = str(text)
    return raw if len(raw) <= max_len else raw[: max_len - 3] + "..."


def _has_openai_key() -> bool:
    """检查 OPENAI_API_KEY 是否存在。"""
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_gemini_key() -> bool:
    """检查 GEMINI_API_KEY / GOOGLE_API_KEY 是否存在（兼容两种命名）。"""
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


# ------------------------------
# HTTP 工具函数（主要用于 Ollama）
# ------------------------------
def _json_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 4.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """发送 JSON HTTP 请求并返回 (json_dict, error_msg)。

    约定：
    - 成功时返回 (dict, None)
    - 失败时返回 (None, "可读错误")

    这样上层逻辑能统一处理错误并写入 fallback_reason / error。
    """
    headers = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url=url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body or "{}"), None
    except urllib.error.HTTPError as e:
        # HTTP 4xx/5xx：优先读取服务端返回体，便于定位问题。
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        return None, f"http {e.code}: {_trim_raw_text(err_body, 300)}"
    except urllib.error.URLError as e:
        # 典型是连接失败、DNS、端口不通。
        return None, f"url error: {e.reason}"
    except Exception as e:
        return None, str(e)


def _get_ollama_base_url() -> str:
    """读取 Ollama base URL（支持环境变量覆盖）。"""
    return (os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL).rstrip("/")


def _get_ollama_model(model: Optional[str] = None) -> str:
    """读取 Ollama 模型名，优先级：函数参数 > 环境变量 > 默认值。"""
    return model or os.getenv("OLLAMA_MODEL") or OLLAMA_DEFAULT_MODEL


def _get_ollama_timeout() -> float:
    """读取 Ollama 请求超时秒数，默认 120 秒。"""
    raw_timeout = (os.getenv("OLLAMA_TIMEOUT") or "120").strip()
    try:
        timeout_val = float(raw_timeout)
    except Exception:
        timeout_val = 120.0
    # 至少 1 秒，避免异常配置导致 0/负数。
    return max(timeout_val, 1.0)


def _has_ollama(base_url: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """检查 Ollama 服务是否可达。

    检查方式：调用 /api/tags。
    返回：
    - (True, None, debug)
    - (False, "ollama not reachable: ...", debug)
    """
    base = (base_url or _get_ollama_base_url()).rstrip("/")
    tags, err = _json_request("GET", f"{base}/api/tags", timeout=2.0)
    if err:
        return False, f"ollama not reachable: {err}", {"base_url": base}
    if not isinstance(tags, dict):
        return False, "ollama not reachable: invalid /api/tags response", {"base_url": base}
    return True, None, {"base_url": base, "model_count": len(tags.get("models", []))}


def _list_ollama_models(base_url: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """获取本地 Ollama 可用模型列表（name 字段）。"""
    base = (base_url or _get_ollama_base_url()).rstrip("/")
    tags, err = _json_request("GET", f"{base}/api/tags", timeout=2.5)
    if err:
        return [], f"ollama not reachable: {err}"

    models = []
    for item in (tags or {}).get("models", []):
        if isinstance(item, dict) and item.get("name"):
            models.append(str(item["name"]))
    return models, None


def _model_matches(requested: str, available: str) -> bool:
    """模型名匹配规则。

    兼容以下场景：
    - requested == available
    - requested='qwen2.5'，available='qwen2.5:3b'
    - requested='qwen2.5:3b'，available='qwen2.5'
    """
    if requested == available:
        return True
    if available.startswith(requested + ":"):
        return True
    if requested.startswith(available + ":"):
        return True
    return False


# ------------------------------
# JSON 解析增强（抗脏输出）
# ------------------------------
def _extract_first_json_object(text: str) -> Optional[str]:
    """从任意文本中提取第一个“平衡大括号”的 JSON object 字符串。

    用于处理模型输出前后混杂解释文字的情况。
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _safe_json_loads(text: str, provider_name: str = "llm") -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """稳健 JSON 解析。

    解析顺序：
    1) 直接 json.loads
    2) 提取 ```json fenced block 再 loads
    3) 提取首个 JSON object 再 loads

    返回：(parsed_dict_or_none, error_or_none, raw_text)
    """
    raw = _trim_raw_text(text)
    content = (text or "").strip()
    if not content:
        return None, f"{provider_name} empty response", raw

    # 1) 直接解析
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed, None, raw
        return {"result": parsed}, None, raw
    except Exception:
        pass

    # 2) fenced code block
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed, None, raw
            return {"result": parsed}, None, raw
        except Exception:
            continue

    # 3) 文本中第一个 JSON object（额外增强，兼容模型在 JSON 前后加说明文字）
    first_obj = _extract_first_json_object(content)
    if first_obj:
        try:
            parsed = json.loads(first_obj)
            if isinstance(parsed, dict):
                return parsed, None, raw
            return {"result": parsed}, None, raw
        except Exception as e:
            return None, f"{provider_name} json parse failed: {e}", raw

    return None, f"{provider_name} json parse failed: no json object found", raw


# ------------------------------
# local fallback payload
# ------------------------------
def _local_explanation_payload(
    lot_features: Dict[str, Any],
    similar_cases: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """本地模板：风险解释。"""
    cases = similar_cases or []
    lot_id = lot_features.get("lot_id", "UNKNOWN")
    return {
        "lot_id": lot_id,
        "risk_summary": (
            f"{lot_id} 呈现电镀高风险模式：E03设备 + Night夜班 + 电流偏高，"
            "与历史异常批次模式相似。"
        ),
        "key_reasons": [
            f"设备: {lot_features.get('machine_id', 'N/A')}（历史异常集中设备）",
            f"班次: {lot_features.get('shift', 'N/A')}（夜班风险偏高）",
            f"电流: {lot_features.get('plating_current_a', 'N/A')}A（相对正常窗口偏高）",
            "模式匹配: 与历史异常 lot 的特征组合接近",
        ],
        "similar_cases": cases[:3],
        "tone": "克制、工程化判断，供工程师复核",
    }


def _local_action_payload(lot_features: Dict[str, Any]) -> Dict[str, Any]:
    """本地模板：建议动作。"""
    return {
        "lot_id": lot_features.get("lot_id", "UNKNOWN"),
        "advice_level": "建议",
        "actions": [
            "对该 lot 执行加严检（增加抽检比例与关键缺陷项复核）",
            "优先检查设备 E03 近期参数漂移与状态记录",
            "复核电流设置与工艺配方，确认未超出控制窗口",
        ],
        "note": "以上为建议动作，不自动执行，需由工程师确认。",
    }


def _build_system_prompt() -> str:
    """统一系统提示词，保持不同 provider 输出风格一致。"""
    return (
        "你是电镀工艺良率分析工程师助手。"
        "不要输出思考过程。"
        "只输出 JSON，不要输出任何额外文字。"
        "请使用工业场景、克制专业语气。"
        "key_reasons 最多 3 条，每条不超过 30 字。"
        "actions 最多 3 条，每条不超过 30 字。"
    )


# ------------------------------
# Provider 调用实现
# ------------------------------
def _call_openai(prompt: str) -> Dict[str, Any]:
    """调用 OpenAI 并返回统一结构。

    返回示例：
    - 成功: {ok: True, provider: 'openai', data: {...}, raw_text: '...'}
    - 失败: {ok: False, provider: 'openai', error: '...'}
    """
    if importlib.util.find_spec("openai") is None:
        return {"ok": False, "provider": "openai", "error": "openai sdk not installed"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"ok": False, "provider": "openai", "error": "openai api key not found"}

    try:
        from openai import OpenAI
    except Exception as e:
        return {"ok": False, "provider": "openai", "error": f"openai import failed: {e}"}

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", OPENAI_DEFAULT_MODEL),
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": prompt},
            ],
        )
        content = (
            (resp.choices[0].message.content if getattr(resp, "choices", None) else "") or ""
        ).strip()

        # 统一走稳健 JSON 解析，避免模型输出夹杂解释文本导致崩溃。
        data, parse_err, raw = _safe_json_loads(content, provider_name="openai")
        if parse_err:
            return {
                "ok": False,
                "provider": "openai",
                "error": parse_err,
                "raw_text": raw,
            }
        return {"ok": True, "provider": "openai", "data": data, "raw_text": raw}
    except Exception as e:
        return {"ok": False, "provider": "openai", "error": f"openai call failed: {e}"}


def _call_gemini(prompt: str) -> Dict[str, Any]:
    """调用 Gemini 并返回统一结构。"""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        return {"ok": False, "provider": "gemini", "error": "gemini api key not found"}

    # 按要求：重点检测 google.genai，而不是笼统检测 google 包。
    if importlib.util.find_spec("google.genai") is None:
        return {"ok": False, "provider": "gemini", "error": "google genai sdk not installed"}

    try:
        from google import genai
    except Exception as e:
        return {"ok": False, "provider": "gemini", "error": f"gemini import failed: {e}"}

    try:
        client = genai.Client(api_key=gemini_key)
        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", GEMINI_DEFAULT_MODEL),
            contents=f"{_build_system_prompt()}\n{prompt}",
        )
        text = (getattr(resp, "text", "") or "").strip()

        data, parse_err, raw = _safe_json_loads(text, provider_name="gemini")
        if parse_err:
            return {
                "ok": False,
                "provider": "gemini",
                "error": parse_err,
                "raw_text": raw,
            }
        return {"ok": True, "provider": "gemini", "data": data, "raw_text": raw}
    except Exception as e:
        return {"ok": False, "provider": "gemini", "error": f"gemini call failed: {e}"}


def _call_ollama(prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """调用 Ollama 原生 API（/api/chat）。"""
    if requests is None:
        return {
            "ok": False,
            "provider": "ollama",
            "error": "requests not installed",
            "debug": {},
        }

    base_url = _get_ollama_base_url()
    chosen_model = _get_ollama_model(model)
    timeout = _get_ollama_timeout()
    chat_url = f"{base_url}/api/chat"

    # 先打印调用调试信息，便于 notebook 现场排查。
    print("OLLAMA DEBUG:")
    print("url:", chat_url)
    print("model:", chosen_model)
    print("timeout:", timeout)

    # 1) 在线检查
    reachable, reach_err, dbg = _has_ollama(base_url=base_url)
    if not reachable:
        return {
            "ok": False,
            "provider": "ollama",
            "error": reach_err or "ollama not reachable",
            "debug": {"base_url": base_url, **dbg},
        }

    # 2) 模型存在性检查
    available_models, list_err = _list_ollama_models(base_url=base_url)
    if list_err:
        return {
            "ok": False,
            "provider": "ollama",
            "error": list_err,
            "debug": {"base_url": base_url},
        }

    if available_models and not any(_model_matches(chosen_model, m) for m in available_models):
        return {
            "ok": False,
            "provider": "ollama",
            "error": "ollama model not found",
            "debug": {
                "base_url": base_url,
                "requested_model": chosen_model,
                "available_models": available_models[:20],
            },
        }

    # 3) 正式调用 /api/chat
    payload = {
        "model": chosen_model,
        # 关键：关闭流式返回，避免 chunk 导致调用方等待超时。
        "stream": False,
        # 降低长时间思考造成的延迟波动。
        "think": False,
        "messages": [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
    }

    try:
        resp = requests.post(
            chat_url,
            json=payload,
            timeout=timeout,
        )
        # 按要求显式抛出 HTTP 异常（4xx/5xx）。
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "provider": "ollama",
            "error": f"ollama call failed: timed out after {timeout} seconds",
            "debug": {"base_url": base_url, "model": chosen_model, "timeout": timeout},
        }
    except requests.exceptions.RequestException as e:
        return {
            "ok": False,
            "provider": "ollama",
            "error": f"ollama call failed: {e}",
            "debug": {"base_url": base_url, "model": chosen_model, "timeout": timeout},
        }
    except Exception as e:
        return {
            "ok": False,
            "provider": "ollama",
            "error": f"ollama call failed: {e}",
            "debug": {"base_url": base_url, "model": chosen_model, "timeout": timeout},
        }

    # 4) 读取内容并解析 JSON
    text = ""
    if isinstance(data, dict):
        text = (((data.get("message") or {}).get("content")) or "").strip()
    if not text:
        return {
            "ok": False,
            "provider": "ollama",
            "error": "ollama empty response",
            "raw_text": _trim_raw_text(data),
            "debug": {"base_url": base_url, "model": chosen_model, "timeout": timeout},
        }

    data, parse_err, raw = _safe_json_loads(text, provider_name="ollama")
    if parse_err:
        return {
            "ok": False,
            "provider": "ollama",
            "error": parse_err,
            "raw_text": raw,
            "debug": {"base_url": base_url, "model": chosen_model, "timeout": timeout},
        }

    return {
        "ok": True,
        "provider": "ollama",
        "data": data,
        "raw_text": raw,
        "debug": {"base_url": base_url, "model": chosen_model, "timeout": timeout},
    }


# ------------------------------
# 输出标准化
# ------------------------------
def _normalize_llm_result(
    payload: Dict[str, Any],
    provider: str,
    attempted_provider: str,
    fallback_used: bool,
    fallback_reason: Optional[str] = None,
    error: Optional[str] = None,
    raw_text: Optional[str] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """给所有返回结果补齐统一调试字段，便于 notebook 直接展示。"""
    result = dict(payload or {})
    result["provider"] = provider
    result["attempted_provider"] = attempted_provider
    result["fallback_used"] = fallback_used
    result["fallback_reason"] = fallback_reason or ""
    result["error"] = error or ""
    result["raw_text"] = _trim_raw_text(raw_text or result.get("raw_text", ""))
    result["debug"] = debug or result.get("debug", {})
    return result


# ------------------------------
# 对外接口：provider 预估
# ------------------------------
def get_llm_provider(provider: str = "auto", use_llm: bool = True) -> str:
    """返回“预估可用 provider”。

    注意：
    - 这是轻量预估，用于 notebook 提示；
    - 最终实际 provider 以 generate_* 返回结果中的 provider 字段为准。
    """
    provider = (provider or "auto").lower().strip()
    if not use_llm:
        return "local"

    if provider == "local":
        return "local"

    if provider == "openai":
        return "openai" if _has_openai_key() else "local"

    if provider == "gemini":
        return "gemini" if _has_gemini_key() else "local"

    if provider == "ollama":
        ok, _, _ = _has_ollama()
        return "ollama" if ok else "local"

    # auto 模式优先级：OpenAI -> Gemini -> Ollama -> local
    if _has_openai_key():
        return "openai"
    if _has_gemini_key():
        return "gemini"
    ok, _, _ = _has_ollama()
    if ok:
        return "ollama"
    return "local"


# ------------------------------
# provider 尝试链路
# ------------------------------
def _build_provider_chain(provider: str, use_llm: bool) -> Tuple[str, List[str]]:
    """构造 provider 尝试链。

    例子：
    - provider='auto'   -> ['openai', 'gemini', 'ollama', 'local']
    - provider='gemini' -> ['gemini', 'openai', 'ollama', 'local']
    - use_llm=False     -> ['local']
    """
    attempted = (provider or "auto").lower().strip()
    if not use_llm:
        return "local", ["local"]

    order = ["openai", "gemini", "ollama", "local"]
    if attempted == "auto":
        return attempted, order
    if attempted == "local":
        return attempted, ["local"]
    if attempted in {"openai", "gemini", "ollama"}:
        # 显式指定：先尝试指定 provider；失败后再按全局顺序回退。
        tail = [p for p in order if p not in {attempted, "local"}] + ["local"]
        return attempted, [attempted] + tail
    return attempted, order


def _call_provider(provider_name: str, prompt: str) -> Dict[str, Any]:
    """按 provider 名称分发调用。"""
    if provider_name == "openai":
        return _call_openai(prompt)
    if provider_name == "gemini":
        return _call_gemini(prompt)
    if provider_name == "ollama":
        return _call_ollama(prompt)
    return {"ok": False, "provider": provider_name, "error": f"unsupported provider: {provider_name}"}


def _run_generation(
    prompt: str,
    local_payload: Dict[str, Any],
    provider: str,
    use_llm: bool,
) -> Dict[str, Any]:
    """统一生成主流程：
    1) 根据 provider/use_llm 构造尝试链
    2) 依次尝试云端/本地 LLM
    3) 全部失败则回退 local 模板，并附带首个失败原因
    """
    attempted, chain = _build_provider_chain(provider=provider, use_llm=use_llm)

    # 明确 local-only 场景：无 fallback。
    if chain == ["local"]:
        return _normalize_llm_result(
            payload=local_payload,
            provider="local",
            attempted_provider=attempted,
            fallback_used=False,
            fallback_reason="",
            error="",
        )

    # 保留“首个失败原因”作为 fallback 解释（最贴近用户意图）。
    first_error = ""
    first_raw = ""
    first_debug: Dict[str, Any] = {}

    for p in chain:
        if p == "local":
            # 已走到最后兜底：返回 local 并携带失败上下文。
            fallback_reason = first_error or "all llm providers unavailable"
            return _normalize_llm_result(
                payload=local_payload,
                provider="local",
                attempted_provider=attempted,
                fallback_used=True,
                fallback_reason=fallback_reason,
                error=first_error,
                raw_text=first_raw,
                debug=first_debug,
            )

        result = _call_provider(p, prompt)
        if result.get("ok"):
            # 是否使用回退：
            # - 显式指定 provider 且最终不是它 -> True
            # - auto 模式且不是第一候选 -> True
            used_fallback = (p != attempted and attempted != "auto") or (attempted == "auto" and p != chain[0])
            return _normalize_llm_result(
                payload=result.get("data") or {},
                provider=p,
                attempted_provider=attempted,
                fallback_used=used_fallback,
                fallback_reason=first_error if used_fallback else "",
                error="",
                raw_text=result.get("raw_text", ""),
                debug=result.get("debug", {}),
            )

        # 仅记录首次失败，避免覆盖最关键的排错信息。
        if not first_error:
            first_error = str(result.get("error", "llm call failed"))
            first_raw = str(result.get("raw_text", ""))
            first_debug = result.get("debug", {}) if isinstance(result.get("debug", {}), dict) else {}

    # 理论上不会走到这里；作为安全兜底保留。
    return _normalize_llm_result(
        payload=local_payload,
        provider="local",
        attempted_provider=attempted,
        fallback_used=True,
        fallback_reason=first_error or "all llm providers unavailable",
        error=first_error,
        raw_text=first_raw,
        debug=first_debug,
    )


# ------------------------------
# 对外接口：风险解释 / 建议动作
# ------------------------------
def generate_lot_explanation(
    lot_features: Dict[str, Any],
    similar_cases: List[Dict[str, Any]] | None = None,
    provider: str = "auto",
    use_llm: bool = True,
) -> Dict[str, Any]:
    """生成 lot 风险解释（支持多 provider + fallback）。"""
    prompt = (
        "请针对以下 lot 输出 JSON。\n"
        "必须字段: lot_id, risk_summary, key_reasons(list), similar_cases(list), tone, actions(list)。\n"
        "不要输出思考过程。只输出 JSON，不要输出任何额外文字。\n"
        "key_reasons 最多3条，每条<=30字；actions 最多3条，每条<=30字。\n"
        f"lot_features={json.dumps(lot_features, ensure_ascii=False)}\n"
        f"similar_cases={json.dumps(similar_cases or [], ensure_ascii=False)}\n"
        "重点强调 E03、Night、电流偏高、与历史异常 lot 相似。"
    )
    local_payload = _local_explanation_payload(lot_features, similar_cases)
    return _run_generation(prompt, local_payload, provider=provider, use_llm=use_llm)


def generate_action_advice(
    lot_features: Dict[str, Any],
    provider: str = "auto",
    use_llm: bool = True,
) -> Dict[str, Any]:
    """生成 lot 建议动作（支持多 provider + fallback）。"""
    prompt = (
        "请针对以下 lot 输出 JSON。\n"
        "必须字段: lot_id, advice_level, actions(list), note, key_reasons(list), risk_summary。\n"
        "不要输出思考过程。只输出 JSON，不要输出任何额外文字。\n"
        "actions 最多3条，每条<=30字；key_reasons 最多3条，每条<=30字。\n"
        f"lot_features={json.dumps(lot_features, ensure_ascii=False)}\n"
        "动作建议需克制、专业，且必须包含：加严检、检查设备E03、复核电流设置。"
    )
    local_payload = _local_action_payload(lot_features)
    return _run_generation(prompt, local_payload, provider=provider, use_llm=use_llm)
