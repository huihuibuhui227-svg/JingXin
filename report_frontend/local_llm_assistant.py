# report_frontend/local_llm_assistant.py
"""严格隔离的按需本地辅助模块（文本可读化摘要 / 缺口追问草稿 / 题库语义检索）。

不可逾越的边界约束：
1. 本模块**绝不参与**任何 L0/L1 指标计算、白盒置信度裁决或维度汇总；
2. 严禁生成任何分数、评级、排名或心理/临床推断；所有本地小模型返回内容必须通过
   `redline_guard.find_assistant_violations()` 二次审查，命中即立刻回退至确定性规则模板；
3. 本地小模型（OpenAI 兼容接口，默认 `http://127.0.0.1:8080/v1`）采用短超时（默认 4s）
   且完全可选：离线或超时时毫秒级（< 5ms）返回确定性结构化摘要或局部 TF-IDF 检索结果，
   绝不拖慢主系统响应速度。
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

import redline_guard

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LLM_MODEL = "Ternary-Bonsai-2-27B"
DEFAULT_LLM_TIMEOUT_SEC = 4.0

ASSISTANT_WATERMARK = (
    "由本地辅助模块生成，仅作文本可读化与追问/检索参考，不参与任何指标计算与评定。"
)


def _llm_config() -> tuple[str, str, float]:
    base = (os.getenv("JX_LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
    model = (os.getenv("JX_LLM_MODEL") or DEFAULT_LLM_MODEL).strip()
    try:
        timeout = max(0.5, min(30.0, float(os.getenv("JX_LLM_TIMEOUT_SEC", str(DEFAULT_LLM_TIMEOUT_SEC)))))
    except ValueError:
        timeout = DEFAULT_LLM_TIMEOUT_SEC
    return base, model, timeout


def _call_local_openai_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    timeout: float | None = None,
    max_tokens: int = 320,
) -> tuple[str | None, str | None, str]:
    """调用本地 OpenAI 兼容小模型（绕过 http_proxy），返回 `(文本, 失败原因, 模型名)`。"""
    base, model, default_timeout = _llm_config()
    eff_timeout = default_timeout if timeout is None else float(timeout)
    url = f"{base}/chat/completions"
    body = json.dumps({
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=eff_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            return None, "本地模型未返回有效 choices", model
        msg = choices[0].get("message") or {}
        content = str(msg.get("content") or "").strip()
        if not content:
            return None, "本地模型返回空内容", model
        return content, None, model
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return None, f"本地模型不可达或超时（{type(exc).__name__}）", model


def _deterministic_report_summary(payload: dict[str, Any]) -> str:
    """毫秒级确定性结构化摘要（零外部依赖、零幻觉、100% 红线安全）。"""
    cov = payload.get("coverage") or {}
    l1 = payload.get("l1") or {}
    dims = payload.get("dimensions") or {}
    lead = l1.get("lead_summary") or {}
    cm = l1.get("cross_modal_coverage") or {}
    sq = l1.get("signal_quality_audit") or {}

    n_passed = cov.get("n_passed", 0)
    n_slots = cov.get("n_slots", 14)
    lines: list[str] = [
        f"1. 证据覆盖概览：本场共有 {n_passed} / {n_slots} 项核心行为指标通过白盒审查。",
    ]
    lead_text = (
        lead.strip()
        if isinstance(lead, str)
        else str((lead or {}).get("headline") or "").strip()
        if isinstance(lead, dict)
        else ""
    )
    if lead_text:
        lines.append(f"2. 结构化导读：{lead_text}")

    dim_bits: list[str] = []
    for _k, d in dims.items():
        name = d.get("name") or _k
        c = d.get("coverage") or {}
        conf = d.get("confidence") or "待核验"
        dim_bits.append(f"{name}（{c.get('n_passed', 0)}/{c.get('n_slots', 0)}，置信度：{conf}）")
    if dim_bits:
        lines.append("3. 分维度证据厚度：" + "；".join(dim_bits) + "。")

    if cm.get("summary_plain"):
        lines.append(f"4. 跨模态时间窗对齐：{cm['summary_plain']}")
    if sq.get("summary_plain"):
        lines.append(f"5. 采集信号质检参考：{sq['summary_plain']}")

    lines.append("6. 审阅边界：所有数值均为客观行为观测记录，不包含任何分数、排序或主观定性结论。")
    return "\n".join(lines)


def _deterministic_interview_guide(payload: dict[str, Any]) -> str:
    """依据本场真实证据缺口与低置信指标，生成 3 条客观人工追问提纲。"""
    dims = payload.get("dimensions") or {}
    l1 = payload.get("l1") or {}
    cm = l1.get("cross_modal_coverage") or {}

    gaps: list[str] = []
    thin_dims: list[str] = []
    for _k, d in dims.items():
        name = d.get("name") or _k
        c = d.get("coverage") or {}
        if (c.get("n_passed", 0) < c.get("n_slots", 0)) or d.get("confidence") in ("不足", "低"):
            thin_dims.append(name)
        for g in (d.get("evidence_gaps") or []):
            if isinstance(g, str) and g.strip():
                gaps.append(f"{name}：{g.strip()}")

    questions: list[str] = []
    if thin_dims:
        questions.append(
            f"1. 针对本场证据较薄的维度（{'、'.join(thin_dims[:2])}），建议在后续面对面交流中"
            "请对方结合具体科研项目实例展开说明关键决策步骤，弥补单场采集片段的覆盖局限。"
        )
    else:
        questions.append(
            "1. 本场五维基础观测覆盖完整，建议结合具体实验复现与异常处理细节开展追问，"
            "由导师依据专业判断核验研究深度。"
        )

    if gaps:
        questions.append(
            f"2. 现场记录存在采集缺口（{gaps[0][:80]}），复核时请区分「设备/环境未采集到」"
            "与「真实行为表现」，必要时安排一次补充演示。"
        )
    else:
        questions.append(
            "2. 可请对方针对最核心的一道开放题口头复述方法假设与验证对照思路，"
            "观察连续阐述过程中的条理性与表达连贯性。"
        )

    if cm.get("status") in ("degraded", "blocked"):
        questions.append(
            f"3. 跨模态时间窗提示（{cm.get('summary_plain', '存在跨模态覆盖偏低')}）："
            "建议不要基于单模态片段做跨通道联动解读，优先核对逐题原始转写与对应录像。"
        )
    else:
        questions.append(
            "3. 建议对照本场逐题离散度图，挑选波动最大的题号回看原始转写，"
            "向对方核实该题的背景熟悉度与思考切入点。"
        )

    return "\n".join(questions)


def _compact_facts_for_prompt(payload: dict[str, Any]) -> str:
    """只把脱敏后的数值事实与缺口喂给本地模型，严禁传入任何个人身份信息。"""
    cov = payload.get("coverage") or {}
    dims = payload.get("dimensions") or {}
    l1 = payload.get("l1") or {}
    facts = {
        "coverage": {"n_passed": cov.get("n_passed"), "n_slots": cov.get("n_slots")},
        "dimensions": [
            {
                "name": d.get("name"),
                "passed": (d.get("coverage") or {}).get("n_passed"),
                "slots": (d.get("coverage") or {}).get("n_slots"),
                "confidence": d.get("confidence"),
                "gaps": (d.get("evidence_gaps") or [])[:3],
            }
            for d in dims.values()
        ],
        "cross_modal": (l1.get("cross_modal_coverage") or {}).get("summary_plain"),
        "signal_quality": (l1.get("signal_quality_audit") or {}).get("summary_plain"),
    }
    return json.dumps(facts, ensure_ascii=False)


def assist_report(
    payload: dict[str, Any],
    *,
    mode: str = "summary",
    use_llm: bool = True,
    timeout: float | None = None,
) -> dict[str, Any]:
    """按需生成脱敏报告的可读化摘要或追问提纲（含本地模型短超时 + 红线过滤 + 确定性回退）。"""
    t0 = time.perf_counter()
    norm_mode = mode if mode in ("summary", "interview_guide") else "summary"
    fallback_text = (
        _deterministic_report_summary(payload)
        if norm_mode == "summary"
        else _deterministic_interview_guide(payload)
    )

    source = "deterministic_fallback"
    model_name = _llm_config()[1]
    fallback_reason: str | None = None
    content = fallback_text

    if use_llm:
        sys_prompt = (
            "你是《衡才智析》系统的本地文本可读化助手。"
            "严格遵守：1) 只复述给定 JSON 里的客观覆盖率、置信度与缺口；"
            "2) 绝对禁止输出任何分数、百分比评分、等级或排名；"
            "3) 绝对禁止做情绪、临床或主观品格推断；"
            "4) 输出简洁中文要点（不超过 220 字）。"
        )
        task_desc = (
            "请把以下脱敏行为证据事实改写为 4 条简洁客观的中文导读："
            if norm_mode == "summary"
            else "请严格依据以下脱敏证据缺口，给出 3 条客观的人工面试补充追问建议："
        )
        user_prompt = f"{task_desc}\n{_compact_facts_for_prompt(payload)}"
        llm_text, err, model_name = _call_local_openai_chat(sys_prompt, user_prompt, timeout=timeout)
        if llm_text:
            violations = redline_guard.find_assistant_violations(llm_text)
            if violations:
                fallback_reason = f"本地模型输出命中红线词（{'、'.join(violations[:3])}），已自动切换确定性合规摘要"
            else:
                content = llm_text
                source = "local_llm"
        else:
            fallback_reason = err or "本地模型未就绪，已使用毫秒级确定性摘要"
    else:
        fallback_reason = "已按请求直接生成毫秒级确定性合规摘要"

    return {
        "mode": norm_mode,
        "content": content,
        "source": source,
        "model": model_name if source == "local_llm" else "deterministic-rule-v1",
        "fallback_reason": fallback_reason,
        "redline_verified": len(redline_guard.find_assistant_violations(content)) == 0,
        "watermark": ASSISTANT_WATERMARK,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def _tokenize_zh_en(text: str) -> list[str]:
    """轻量中英双语分词（英文词 + 中文单字与双字切片），零外部分词库依赖。"""
    s = (text or "").lower().strip()
    if not s:
        return []
    tokens: list[str] = re.findall(r"[a-z0-9_]+", s)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", s)
    for run in cjk_runs:
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
    return tokens


def search_question_bank(
    module: str,
    query: str,
    *,
    top_k: int = 5,
    use_llm: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    """本地毫秒级题库 TF-IDF 检索（可选本地小模型扩展检索词）。"""
    import question_bank

    t0 = time.perf_counter()
    q_raw = (query or "").strip()
    if not q_raw:
        return {
            "module": module,
            "query": "",
            "expanded_query": "",
            "source": "local_tfidf",
            "hits": [],
            "watermark": ASSISTANT_WATERMARK,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    state = question_bank.load(module)
    docs: list[dict[str, Any]] = []
    # 内置默认题库
    for idx, q_text in enumerate(state.get("builtin") or []):
        text = q_text if isinstance(q_text, str) else str(q_text.get("question") or "")
        docs.append({
            "bank_id": "builtin",
            "bank_name": "系统内置默认题库",
            "index": idx,
            "question": text,
            "reference_answer": "",
            "is_active": state.get("active_id") is None,
        })
    # 自定义题库
    for b in state.get("banks") or []:
        bid = b.get("id")
        bname = b.get("name") or str(bid)
        for idx, item in enumerate(b.get("questions") or []):
            if isinstance(item, str):
                q_t, r_t = item, ""
            else:
                q_t = str(item.get("question") or "")
                r_t = str(item.get("reference_answer") or "")
            docs.append({
                "bank_id": bid,
                "bank_name": bname,
                "index": idx,
                "question": q_t,
                "reference_answer": r_t,
                "is_active": state.get("active_id") == bid,
            })

    expanded = q_raw
    source = "local_tfidf"
    if use_llm:
        llm_kw, _err, _mdl = _call_local_openai_chat(
            "请为给定科研面试检索意图补充 3-5 个简短中文同义检索词，只返回空格分隔的词语，不解释。",
            q_raw,
            timeout=timeout,
            max_tokens=48,
        )
        if llm_kw and not redline_guard.find_assistant_violations(llm_kw):
            expanded = f"{q_raw} {llm_kw}"
            source = "local_llm_expanded_tfidf"

    q_tokens = _tokenize_zh_en(expanded)
    q_set = set(q_tokens)
    doc_tokens = [_tokenize_zh_en(f"{d['question']} {d['reference_answer']}") for d in docs]
    n_docs = max(len(docs), 1)
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1

    def _tfidf_vec(toks: list[str]) -> dict[str, float]:
        tf: dict[str, int] = {}
        for tok in toks:
            tf[tok] = tf.get(tok, 0) + 1
        vec: dict[str, float] = {}
        for tok, cnt in tf.items():
            idf = math.log((n_docs + 1.0) / (df.get(tok, 0) + 1.0)) + 1.0
            vec[tok] = (1.0 + math.log(cnt)) * idf
        return vec

    q_vec = _tfidf_vec(q_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scored: list[dict[str, Any]] = []
    for d, toks in zip(docs, doc_tokens):
        d_vec = _tfidf_vec(toks)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec[t] * d_vec[t] for t in q_vec if t in d_vec)
        sim = dot / (q_norm * d_norm)
        if sim > 0.0:
            matched = sorted(
                {t for t in q_set if t in d_vec and len(t) >= 2},
                key=len,
                reverse=True,
            )[:6]
            scored.append({
                **d,
                "similarity": round(float(sim), 4),
                "matched_terms": matched,
            })

    scored.sort(key=lambda x: (x["similarity"], 1 if x["is_active"] else 0), reverse=True)
    return {
        "module": module,
        "query": q_raw,
        "expanded_query": expanded,
        "source": source,
        "total_candidates": len(docs),
        "hits": scored[: max(1, min(int(top_k), 20))],
        "watermark": ASSISTANT_WATERMARK,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
