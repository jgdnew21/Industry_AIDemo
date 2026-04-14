"""LLM provider wrapper for AAMI yield demo.

Priority for provider='auto': OpenAI > Gemini > Ollama > Local fallback.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"
OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def _load_env_from_dotenv() -> None:
    """Best-effort load for .env so API keys can be managed outside code."""
    if importlib.util.find_spec("dotenv") is None:
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:
        # Keep demo stable even if dotenv behaves unexpectedly.
        return


_load_env_from_dotenv()


def _trim_raw_text(text: Any, max_len: int = 1200) -> str:
    if text is None:
        return ""
    raw = str(text)
    return raw if len(raw) <= max_len else raw[: max_len - 3] + "..."


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _json_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 4.0,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        return None, f"http {e.code}: {_trim_raw_text(err_body, 300)}"
    except urllib.error.URLError as e:
        return None, f"url error: {e.reason}"
    except Exception as e:
        return None, str(e)


def _get_ollama_base_url() -> str:
    return (os.getenv("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL).rstrip("/")


def _get_ollama_model(model: Optional[str] = None) -> str:
    return model or os.getenv("OLLAMA_MODEL") or OLLAMA_DEFAULT_MODEL


def _has_ollama(base_url: Optional[str] = None) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    base = (base_url or _get_ollama_base_url()).rstrip("/")
    tags, err = _json_request("GET", f"{base}/api/tags", timeout=2.0)
    if err:
        return False, f"ollama not reachable: {err}", {"base_url": base}
    if not isinstance(tags, dict):
        return False, "ollama not reachable: invalid /api/tags response", {"base_url": base}
    return True, None, {"base_url": base, "model_count": len(tags.get("models", []))}


def _list_ollama_models(base_url: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
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
    if requested == available:
        return True
    if available.startswith(requested + ":"):
        return True
    if requested.startswith(available + ":"):
        return True
    return False


def _extract_first_json_object(text: str) -> Optional[str]:
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


def _safe_json_loads(text: str, provider_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    raw = _trim_raw_text(text)
    content = (text or "").strip()
    if not content:
        return None, f"{provider_name} empty response", raw

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed, None, raw
        return {"result": parsed}, None, raw
    except Exception:
        pass

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, dict):
                return parsed, None, raw
            return {"result": parsed}, None, raw
        except Exception:
            continue

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


def _local_explanation_payload(
    lot_features: Dict[str, Any],
    similar_cases: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
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
    return (
        "你是电镀工艺良率分析工程师助手。请使用工业场景、克制专业语气，"
        "输出结构化 JSON，仅给出原因解释和建议动作，不夸张。"
    )


def _call_openai(prompt: str) -> Dict[str, Any]:
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
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        return {"ok": False, "provider": "gemini", "error": "gemini api key not found"}

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
    base_url = _get_ollama_base_url()
    chosen_model = _get_ollama_model(model)

    reachable, reach_err, dbg = _has_ollama(base_url=base_url)
    if not reachable:
        return {
            "ok": False,
            "provider": "ollama",
            "error": reach_err or "ollama not reachable",
            "debug": {"base_url": base_url, **dbg},
        }

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

    payload = {
        "model": chosen_model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.2},
    }

    resp, err = _json_request("POST", f"{base_url}/api/chat", payload=payload, timeout=30.0)
    if err:
        return {
            "ok": False,
            "provider": "ollama",
            "error": f"ollama call failed: {err}",
            "debug": {"base_url": base_url, "model": chosen_model},
        }

    text = ""
    if isinstance(resp, dict):
        text = (((resp.get("message") or {}).get("content")) or "").strip()
    if not text:
        return {
            "ok": False,
            "provider": "ollama",
            "error": "ollama empty response",
            "raw_text": _trim_raw_text(resp),
            "debug": {"base_url": base_url, "model": chosen_model},
        }

    data, parse_err, raw = _safe_json_loads(text, provider_name="ollama")
    if parse_err:
        return {
            "ok": False,
            "provider": "ollama",
            "error": parse_err,
            "raw_text": raw,
            "debug": {"base_url": base_url, "model": chosen_model},
        }

    return {
        "ok": True,
        "provider": "ollama",
        "data": data,
        "raw_text": raw,
        "debug": {"base_url": base_url, "model": chosen_model},
    }


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
    result = dict(payload or {})
    result["provider"] = provider
    result["attempted_provider"] = attempted_provider
    result["fallback_used"] = fallback_used
    result["fallback_reason"] = fallback_reason or ""
    result["error"] = error or ""
    result["raw_text"] = _trim_raw_text(raw_text or result.get("raw_text", ""))
    result["debug"] = debug or result.get("debug", {})
    return result


def get_llm_provider(provider: str = "auto", use_llm: bool = True) -> str:
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

    if _has_openai_key():
        return "openai"
    if _has_gemini_key():
        return "gemini"
    ok, _, _ = _has_ollama()
    if ok:
        return "ollama"
    return "local"


def _build_provider_chain(provider: str, use_llm: bool) -> Tuple[str, List[str]]:
    attempted = (provider or "auto").lower().strip()
    if not use_llm:
        return "local", ["local"]

    order = ["openai", "gemini", "ollama", "local"]
    if attempted == "auto":
        return attempted, order
    if attempted == "local":
        return attempted, ["local"]
    if attempted in {"openai", "gemini", "ollama"}:
        tail = [p for p in order if p not in {attempted, "local"}] + ["local"]
        return attempted, [attempted] + tail
    return attempted, order


def _call_provider(provider_name: str, prompt: str) -> Dict[str, Any]:
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
    attempted, chain = _build_provider_chain(provider=provider, use_llm=use_llm)

    if chain == ["local"]:
        return _normalize_llm_result(
            payload=local_payload,
            provider="local",
            attempted_provider=attempted,
            fallback_used=False,
            fallback_reason="",
            error="",
        )

    first_error = ""
    first_raw = ""
    first_debug: Dict[str, Any] = {}

    for p in chain:
        if p == "local":
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

        if not first_error:
            first_error = str(result.get("error", "llm call failed"))
            first_raw = str(result.get("raw_text", ""))
            first_debug = result.get("debug", {}) if isinstance(result.get("debug", {}), dict) else {}

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


def generate_lot_explanation(
    lot_features: Dict[str, Any],
    similar_cases: List[Dict[str, Any]] | None = None,
    provider: str = "auto",
    use_llm: bool = True,
) -> Dict[str, Any]:
    prompt = (
        "请针对以下 lot 输出 JSON，字段包含: lot_id, risk_summary, key_reasons(list), "
        "similar_cases(list), tone。\n"
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
    prompt = (
        "请针对以下 lot 输出 JSON，字段包含: lot_id, advice_level, actions(list), note。\n"
        f"lot_features={json.dumps(lot_features, ensure_ascii=False)}\n"
        "动作建议需克制、专业，且必须包含：加严检、检查设备E03、复核电流设置。"
    )
    local_payload = _local_action_payload(lot_features)
    return _run_generation(prompt, local_payload, provider=provider, use_llm=use_llm)
