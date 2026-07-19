"""Prompt and request audit helpers for CERA."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from certa.reproducibility.canonical_json import canonical_json, canonical_json_hash, canonical_text_hash


def _stable_json(value: Any) -> str:
    return canonical_json(value)


def stable_hash_text(text: str, n: int = 16) -> str:
    return canonical_text_hash(str(text or ""), n=n)


def stable_hash_json(value: Any, n: int = 16) -> str:
    return canonical_json_hash(value, n=n)


def _count_tokens(generator: Any, text: str) -> int:
    if generator is not None and hasattr(generator, "count_generation_prompt_tokens"):
        try:
            return int(generator.count_generation_prompt_tokens(text))
        except Exception:
            pass
    return max(1, (len(text or "") + 3) // 4) if text else 0


def _get_from_mapping(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return default


def build_cera_request_audit(
    *,
    prompt: str,
    packet: Any,
    query_contract: Any,
    args: Any = None,
    generator: Any = None,
    generation_output: Optional[Mapping[str, Any]] = None,
    generation_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    packet_payload = packet.to_dict() if hasattr(packet, "to_dict") else packet
    query_payload = query_contract.to_dict() if hasattr(query_contract, "to_dict") else query_contract
    packet_hash = _get_from_mapping(_get_from_mapping(packet_payload, "metadata", {}), "packet_hash")
    packet_hash = packet_hash or stable_hash_json(packet_payload)
    query_contract_hash = stable_hash_json(query_payload)
    sampling = {
        "max_tokens": getattr(args, "cera_max_tokens", None),
        "temperature": getattr(args, "cera_temperature", None),
        "top_p": getattr(args, "top_p", None),
    }
    input_tokens = int(_get_from_mapping(generation_output or {}, "input_token_count", 0) or 0)
    if not input_tokens:
        input_tokens = _count_tokens(generator, prompt)
    output_tokens = int(_get_from_mapping(generation_output or {}, "generated_token_count", 0) or 0)
    audit = {
        "packet_hash": packet_hash,
        "query_contract_hash": query_contract_hash,
        "prompt_hash": stable_hash_text(prompt),
        "model": str(
            _get_from_mapping(generation_output or {}, "api_model", "")
            or getattr(generator, "model", "")
            or getattr(args, "api_model", "")
            or getattr(args, "model_path", "")
        ),
        "backend": str(
            _get_from_mapping(generation_output or {}, "generator_backend", "")
            or getattr(generator, "backend_name", "")
            or getattr(args, "generator_backend", "")
        ),
        "api_base_url": str(
            _get_from_mapping(generation_output or {}, "api_base_url", "")
            or getattr(generator, "api_base_url", "")
            or getattr(args, "api_base_url", "")
        ),
        "sampling": sampling,
        "api_cache_hit": bool(_get_from_mapping(generation_output or {}, "api_cache_hit", False)),
        "api_cache_mode": str(_get_from_mapping(generation_output or {}, "api_cache_mode", "") or getattr(args, "api_cache_mode", "")),
        "latency_seconds": float(
            generation_seconds
            if generation_seconds is not None
            else _get_from_mapping(generation_output or {}, "generation_seconds", 0.0)
            or 0.0
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    request_payload = {
        "packet_hash": audit["packet_hash"],
        "query_contract_hash": audit["query_contract_hash"],
        "prompt_hash": audit["prompt_hash"],
        "model": audit["model"],
        "backend": audit["backend"],
        "api_base_url": audit["api_base_url"],
        "sampling": audit["sampling"],
    }
    audit["request_hash"] = stable_hash_json(request_payload)
    return audit
