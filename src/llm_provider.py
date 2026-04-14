"""LLM provider wrapper for AAMI yield demo.

Priority: OpenAI > Gemini > Local fallback.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List


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


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def get_llm_provider(provider: str = "auto", use_llm: bool = True) -> str:
    provider = (provider or "auto").lower().strip()
    if not use_llm:
        return "local"

    if provider in {"openai", "gemini", "local"}:
        if provider == "openai" and _has_openai_key():
            return "openai"
        if provider == "gemini" and _has_gemini_key():
            return "gemini"
        if provider == "local":
            return "local"
        return "local"

    if _has_openai_key():
        return "openai"
    if _has_gemini_key():
        return "gemini"
    return "local"


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


def _call_openai(prompt: str) -> Dict[str, Any]:
    if importlib.util.find_spec("openai") is None:
        return {"error": "openai sdk not installed"}

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是电镀工艺良率分析工程师助手。请使用工业场景、克制专业语气，"
                        "输出结构化 JSON，仅给出原因解释和建议动作，不夸张。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        return {"error": f"openai call failed: {e}"}


def _call_gemini(prompt: str) -> Dict[str, Any]:
    if importlib.util.find_spec("google") is None and importlib.util.find_spec("google.genai") is None:
        return {"error": "google genai sdk not installed"}

    from google import genai

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=key)
    try:
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                "你是电镀工艺良率分析工程师助手。请使用工业场景、克制专业语气，"
                "输出结构化 JSON，仅给出原因解释和建议动作，不夸张。\n"
                + prompt
            ),
        )
        text = getattr(resp, "text", "") or "{}"
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(text)
    except Exception as e:
        return {"error": f"gemini call failed: {e}"}


def generate_lot_explanation(
    lot_features: Dict[str, Any],
    similar_cases: List[Dict[str, Any]] | None = None,
    provider: str = "auto",
    use_llm: bool = True,
) -> Dict[str, Any]:
    resolved = get_llm_provider(provider=provider, use_llm=use_llm)
    local_payload = _local_explanation_payload(lot_features, similar_cases)

    prompt = (
        "请针对以下 lot 输出 JSON，字段包含: lot_id, risk_summary, key_reasons(list), "
        "similar_cases(list), tone。\n"
        f"lot_features={json.dumps(lot_features, ensure_ascii=False)}\n"
        f"similar_cases={json.dumps(similar_cases or [], ensure_ascii=False)}\n"
        "重点强调 E03、Night、电流偏高、与历史异常 lot 相似。"
    )

    if resolved == "openai":
        payload = _call_openai(prompt)
        if "error" not in payload:
            payload["provider"] = "openai"
            return payload
    elif resolved == "gemini":
        payload = _call_gemini(prompt)
        if "error" not in payload:
            payload["provider"] = "gemini"
            return payload

    local_payload["provider"] = "local"
    return local_payload


def generate_action_advice(
    lot_features: Dict[str, Any],
    provider: str = "auto",
    use_llm: bool = True,
) -> Dict[str, Any]:
    resolved = get_llm_provider(provider=provider, use_llm=use_llm)
    local_payload = _local_action_payload(lot_features)

    prompt = (
        "请针对以下 lot 输出 JSON，字段包含: lot_id, advice_level, actions(list), note。\n"
        f"lot_features={json.dumps(lot_features, ensure_ascii=False)}\n"
        "动作建议需克制、专业，且必须包含：加严检、检查设备E03、复核电流设置。"
    )

    if resolved == "openai":
        payload = _call_openai(prompt)
        if "error" not in payload:
            payload["provider"] = "openai"
            return payload
    elif resolved == "gemini":
        payload = _call_gemini(prompt)
        if "error" not in payload:
            payload["provider"] = "gemini"
            return payload

    local_payload["provider"] = "local"
    return local_payload
