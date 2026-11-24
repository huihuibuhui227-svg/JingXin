# mcp_career/server.py
"""标准 JSON-RPC 2.0 Career MCP Server（零外部网络/爬虫依赖，严格不打分）。

提供 4 个符合 Model Context Protocol 规范的确定性工具：
1. `search_career_positions`    — 检索校招/科研助理岗位与可观测能力画像
2. `decompose_jd_competencies`  — 将任意 JD 拆解为「可观测行为证据清单」与「显式封停硬技能」
3. `generate_ai_critique_task`  — 为目标岗位生成「阅读 AI 回答 · 找错审辨 · 阐述修正」三阶段实训题
4. `align_session_evidence_to_jd` — 将真实会话实测证据对齐至岗位要求，仅披露覆盖与缺口，严禁给分

支持以 `python -m mcp_career.server` 标准 stdio 方式运行，也可直接通过
`handle_jsonrpc(request_obj)` 被 Flask `app.py` 桥接路由调用。
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import redline_guard

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "hengcai-career-mcp",
    "version": "1.0.0",
    "title": "衡才智析 · AI 元能力与求职证据对照 MCP 服务",
}

CATALOG_PATH = Path(__file__).resolve().with_name("positions_catalog.json")

_JD_DIMENSION_RULES: tuple[dict[str, Any], ...] = (
    {
        "key": "structured_reasoning",
        "label": "结构化逻辑与因果阐述",
        "theoretical_construct": "Schraw & Dennison (1994) · 认知规划与条理阐述",
        "keywords": ("结构", "逻辑", "汇报", "沟通", "阐述", "表达", "拆解", "条理", "方案"),
        "indicators": ["discourse_marker_rate", "answer_detail_chars_p50", "speech_rate"],
        "advice": "阐述方案时显式使用因果与转折连接标记，先给核心前提再展开论据细节。",
    },
    {
        "key": "ai_premise_critique",
        "label": "AI 输出审辨与边界修正",
        "theoretical_construct": "Tankelevitch et al. (CHI 2024) & Lee et al. (CHI 2025) · 监测、核验与排错三环",
        "keywords": ("大模型", "AI", "生成", "幻觉", "审辨", "批判", "假设", "验证", "归因", "口径", "漂移", "降级", "审计", "合规"),
        "indicators": [
            "counter_premise_marker_rate",
            "evidence_verification_marker_rate",
            "revision_anchor_density",
            "metacognitive_regulation_balance",
        ],
        "advice": "刻意强化「指出漏洞前提 → 查证数据口径 → 给出具体修正方案」三环闭环表述。",
    },
    {
        "key": "vocal_cadence",
        "label": "陈述节奏与停顿控制",
        "theoretical_construct": "Nelson & Narens (1990) · 表达过程监测与节律调控",
        "keywords": ("节奏", "汇报", "表达", "追问", "答辩", "稳定", "讲授", "沟通"),
        "indicators": ["pause_structure_entropy", "voiced_ratio", "pitch_variability_per_sec", "response_latency_sec"],
        "advice": "面对追问时控制首句组织反应时长，在结论转折处保留短暂均匀停顿。",
    },
    {
        "key": "nonverbal_poise",
        "label": "注视稳定与上肢协同表达",
        "theoretical_construct": "多模态沟通协同 · 视线驻留与姿态稳态",
        "keywords": ("宣讲", "展示", "演示", "讲师", "产品", "协调", "面对", "汇报"),
        "indicators": ["gaze_stability", "gaze_dwell_p50", "gesture_activity_per_sec", "hand_openness_p50"],
        "advice": "汇报关键数据或系统结构时保持摄像头正向注视与上肢自然进入画面区域。",
    },
)

_BLOCKED_JD_SKILL_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("编程语言上机实操、源码级调试与工程产线代码质量", ("python", "c++", "java", "go", "rust", "cuda", "sql", "typescript", "react", "编程", "编码", "代码", "算法题", "开发")),
    ("历史项目商业收益或学术论文同行评审原件核验", ("项目经验", "实习经历", "落地", "营收", "论文", "一作", "专利", "上线")),
    ("书面笔试、专业资格证书与跨团队长期协同背调", ("证书", "笔试", "资质", "背调", "等保", "全栈实战")),
)


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    with open(CATALOG_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.get("positions") or [])


def _guard_text(text: str) -> str:
    """过滤或拦截任何越过红线规则的字段内容，确保零打分/零临床措辞。"""
    hits = redline_guard.find_assistant_violations(text)
    if hits:
        return (
            "该表述涉及未开放评价或敏感词汇已被红线守卫自动拦截；"
            "本服务仅披露可观测行为证据与缺口。"
        )
    return text


def _get_position(position_id: str) -> dict[str, Any] | None:
    pid = str(position_id or "").strip()
    if not pid:
        return None
    for p in load_catalog():
        if p.get("position_id") == pid:
            return p
    return None


def tool_search_career_positions(
    keyword: str = "",
    track: str = "",
    level: str = "",
) -> dict[str, Any]:
    kw = str(keyword or "").strip().lower()
    tr = str(track or "").strip()
    lv = str(level or "").strip()
    positions = load_catalog()
    matched = []
    for p in positions:
        if tr and tr != "全部" and p.get("track") != tr:
            continue
        if lv and lv != "全部" and p.get("level") != lv:
            continue
        if kw:
            hay = " ".join([
                str(p.get("title", "")),
                str(p.get("track", "")),
                str(p.get("jd_summary", "")),
                " ".join(c.get("label", "") for c in (p.get("observable_competencies") or [])),
            ]).lower()
            if kw not in hay:
                continue
        matched.append(p)
    tracks = sorted({str(p.get("track") or "") for p in positions if p.get("track")})
    levels = sorted({str(p.get("level") or "") for p in positions if p.get("level")})
    return {
        "total": len(matched),
        "tracks": tracks,
        "levels": levels,
        "positions": matched,
        "scoring_participation": False,
    }


def tool_decompose_jd_competencies(jd_text: str) -> dict[str, Any]:
    raw = str(jd_text or "").strip()
    if not raw:
        raise ValueError("请输入岗位职责或 JD 文本后再拆解。")
    lower = raw.lower()
    observable: list[dict[str, Any]] = []
    for rule in _JD_DIMENSION_RULES:
        hits = [kw for kw in rule["keywords"] if kw.lower() in lower]
        if hits:
            observable.append({
                "key": rule["key"],
                "label": rule["label"],
                "theoretical_construct": rule.get("theoretical_construct", ""),
                "matched_keywords": hits,
                "required_indicators": list(rule["indicators"]),
                "critique_focus": rule["advice"],
            })
    if not observable:
        observable.append({
            "key": _JD_DIMENSION_RULES[0]["key"],
            "label": _JD_DIMENSION_RULES[0]["label"],
            "theoretical_construct": _JD_DIMENSION_RULES[0].get("theoretical_construct", ""),
            "matched_keywords": ["通用表达前提"],
            "required_indicators": list(_JD_DIMENSION_RULES[0]["indicators"]),
            "critique_focus": _JD_DIMENSION_RULES[0]["advice"],
        })
        observable.append({
            "key": _JD_DIMENSION_RULES[1]["key"],
            "label": _JD_DIMENSION_RULES[1]["label"],
            "theoretical_construct": _JD_DIMENSION_RULES[1].get("theoretical_construct", ""),
            "matched_keywords": ["通用审辨边界"],
            "required_indicators": list(_JD_DIMENSION_RULES[1]["indicators"]),
            "critique_focus": _JD_DIMENSION_RULES[1]["advice"],
        })

    blocked: list[str] = []
    for label, kws in _BLOCKED_JD_SKILL_RULES:
        if any(kw.lower() in lower for kw in kws):
            blocked.append(label)
    if not blocked:
        blocked.append("岗位涉及的线下专业笔试及工程源码深度审核不在行为采录观测范围内")

    return {
        "source": "custom_jd",
        "char_count": len(raw),
        "theory_framework": "Nelson & Narens (1990) / Tankelevitch et al. (CHI 2024) / Lee et al. (CHI 2025)",
        "observable_competencies": observable,
        "blocked_hard_skills": blocked,
        "non_scoring_notice": "岗位拆解仅区分可观测行为线索与不可测硬技能缺口，不生成岗前分值。",
    }


def tool_generate_ai_critique_task(
    position_id: str = "",
    jd_text: str = "",
) -> dict[str, Any]:
    pos = _get_position(position_id)
    if pos is not None:
        title = str(pos.get("title") or "目标岗位")
        base_prompt = str(pos.get("sample_ai_critique_prompt") or "")
        focus_items = [str(c.get("critique_focus") or "") for c in (pos.get("observable_competencies") or [])]
    else:
        decomposed = tool_decompose_jd_competencies(
            jd_text or "负责结构化方案表达、统计口径核验与模型边界审辨。"
        )
        title = "自定义岗位 JD 审辨实训"
        first_focus = decomposed["observable_competencies"][0]["label"]
        base_prompt = (
            f"面向该岗位的「{first_focus}」要求，AI 给出了一份看似流利但省略关键约束的前提摘要："
            "「只需采用单一均值聚合结果并对缺失值自动补零，即可稳健得出最终决定」。"
            "请依次阐述：① 该前提为何容易失真；② 列举典型边界反例；③ 给出你的可核验修正做法。"
        )
        focus_items = [str(c.get("critique_focus") or "") for c in decomposed["observable_competencies"]]

    return {
        "position_title": title,
        "task_pattern": "read_ai -> critique -> revise",
        "stages": [
            {"stage": "stage_read_ai", "label": "阶段一 · 阅读核查 AI 回答", "tip": "快速定位论证前提、数据口径或条件边界中的隐蔽漏洞"},
            {"stage": "stage_critique", "label": "阶段二 · 指出前提漏洞与反例", "tip": "显式使用前提质疑与边界限定连接词说明为何不成立"},
            {"stage": "stage_revise", "label": "阶段三 · 阐述修正与核验方案", "tip": "按「指出缺陷 → 替代方案 → 验证条件」给出闭环阐述"},
        ],
        "question_prompt": _guard_text(base_prompt),
        "focus_checklist": [ _guard_text(f) for f in focus_items if f ],
    }


def tool_align_session_evidence_to_jd(
    session_id: str,
    position_id: str = "",
    jd_text: str = "",
    *,
    logs: dict | None = None,
) -> dict[str, Any]:
    from report_frontend.data_loader import LogDataLoader
    from report_frontend.l1_report import build_l1_report

    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("缺少 session_id，无法执行会话证据对照。")

    pos = _get_position(position_id)
    if pos is not None:
        role_title = str(pos.get("title") or "")
        competencies = list(pos.get("observable_competencies") or [])
        blocked_skills = list(pos.get("blocked_hard_skills") or [])
    else:
        decomposed = tool_decompose_jd_competencies(
            jd_text or "要求具备清晰结构化表达与 AI 结果审辨能力。"
        )
        role_title = "自定义岗位 JD"
        competencies = list(decomposed["observable_competencies"])
        blocked_skills = list(decomposed["blocked_hard_skills"])

    if logs is None:
        loader = LogDataLoader()
        loaded = loader.get_fused_latest_data(sid) or {}
        actual_sid = loader.target_session or sid
    else:
        loaded = logs
        actual_sid = sid

    payload = build_l1_report(actual_sid, logs=loaded, scene_profile="ai_metacognition")

    ind_meta = (payload.get("l1") or {}).get("indicator_meta") or {}
    ind_map: dict[str, dict[str, Any]] = {}
    for row in (payload.get("l1") or {}).get("indicators") or []:
        if isinstance(row, dict) and row.get("name"):
            ind_map[row["name"]] = row

    ext_all = (payload.get("scene_profile") or {}).get("extensions") or {}
    for block_key in ("metacognition_text", "metacognition_regulation"):
        block = ext_all.get(block_key) or {}
        for row in block.get("indicators") or []:
            if isinstance(row, dict) and row.get("name"):
                ind_map[row["name"]] = row

    alignments: list[dict[str, Any]] = []
    n_covered = 0
    n_partial = 0
    n_missing = 0

    for comp in competencies:
        req_keys = list(comp.get("required_indicators") or [])
        evidences: list[dict[str, Any]] = []
        ok_count = 0
        high_mid_count = 0
        for key in req_keys:
            item = ind_map.get(key)
            if item is None:
                evidences.append({
                    "indicator": key,
                    "label": key,
                    "value": None,
                    "confidence": "无",
                    "gap": "not_produced",
                })
                continue
            val = item.get("value")
            conf = str(item.get("confidence") or "无")
            gap = item.get("gap")
            if val is not None:
                ok_count += 1
                if conf in ("高", "中"):
                    high_mid_count += 1
            meta_lbl = (ind_meta.get(key) or {}).get("label") if isinstance(ind_meta.get(key), dict) else None
            raw_lbl = str(meta_lbl or item.get("label") or item.get("formula") or key)
            clean_lbl = raw_lbl.replace("**", "").replace("`", "").strip()
            evidences.append({
                "indicator": key,
                "label": clean_lbl,
                "value": val,
                "unit": item.get("unit") or "",
                "confidence": conf,
                "gap": gap,
            })

        if ok_count == len(req_keys) and high_mid_count >= max(1, len(req_keys) - 1):
            cov_status = "高/中置信证据充分"
            n_covered += 1
        elif ok_count > 0:
            cov_status = "部分证据可查（待补充样本）"
            n_partial += 1
        else:
            cov_status = "缺口待补录"
            n_missing += 1

        alignments.append({
            "key": comp.get("key"),
            "label": comp.get("label"),
            "theoretical_construct": _guard_text(str(comp.get("theoretical_construct") or "")),
            "coverage_status": cov_status,
            "critique_focus": _guard_text(str(comp.get("critique_focus") or "")),
            "evidences": evidences,
        })

    referee = (payload.get("l1") or {}).get("referee_ml") or {}
    gate_info = referee.get("evidence_gate") or {}

    return {
        "session_id": actual_sid,
        "position_id": position_id or "custom_jd",
        "position_title": role_title,
        "scoring_participation": False,
        "theory_framework": "人机协同元认知监测—核验—排错三环框架（CHI 2024/2025；Nelson & Narens 1990）",
        "non_scoring_statement": "本对照仅核查岗位可观测需求在本场多模态录测中的证据覆盖与缺失，不出具契合度分数或筛序结论。",
        "summary_counts": {
            "competencies_total": len(competencies),
            "covered_strong": n_covered,
            "covered_partial": n_partial,
            "missing_gap": n_missing,
        },
        "referee_stage_summary": {
            "stage_label": gate_info.get("stage_label") or "基础核验",
            "uncertainty_escalated": bool(gate_info.get("uncertainty_escalated")),
        },
        "competency_alignments": alignments,
        "explicit_blocked_hard_skills": blocked_skills,
        "next_practice_task": tool_generate_ai_critique_task(position_id=position_id, jd_text=jd_text),
    }


TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "name": "search_career_positions",
        "description": "按关键词、赛道与层级检索校招及科研助理岗位画像与可观测行为要求。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "岗位关键词，如 算法、数据、科研、后端、金融"},
                "track": {"type": "string", "description": "所属赛道"},
                "level": {"type": "string", "description": "岗位层级：校招/科研助理/实习"},
            },
        },
    },
    {
        "name": "decompose_jd_competencies",
        "description": "把任意岗位 JD 文本拆解为可观测行为线索清单与显式封停的不可测硬技能。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "jd_text": {"type": "string", "description": "岗位描述文本"},
            },
            "required": ["jd_text"],
        },
    },
    {
        "name": "generate_ai_critique_task",
        "description": "结合目标岗位生成「阅读 AI 回答 → 指出漏洞 → 口头修正」三阶段 AI 审辨实训题。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "position_id": {"type": "string", "description": "内置岗位编号，可空"},
                "jd_text": {"type": "string", "description": "当不选内置岗位时的自定义 JD 文本"},
            },
        },
    },
    {
        "name": "align_session_evidence_to_jd",
        "description": "读取指定会话的多模态实测与文本审辨证据，生成面向目标岗位的覆盖与缺口对照清单（严禁打分）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "已录制会话编号"},
                "position_id": {"type": "string", "description": "内置岗位编号"},
                "jd_text": {"type": "string", "description": "自定义 JD 文本（position_id 为空时生效）"},
            },
            "required": ["session_id"],
        },
    },
]


class CareerMcpServer:
    """无状态标准 JSON-RPC 2.0 MCP 服务调度器。"""

    def dispatch_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = dict(arguments or {})
        if name == "search_career_positions":
            return tool_search_career_positions(
                keyword=args.get("keyword", ""),
                track=args.get("track", ""),
                level=args.get("level", ""),
            )
        if name == "decompose_jd_competencies":
            return tool_decompose_jd_competencies(jd_text=args.get("jd_text", ""))
        if name == "generate_ai_critique_task":
            return tool_generate_ai_critique_task(
                position_id=args.get("position_id", ""),
                jd_text=args.get("jd_text", ""),
            )
        if name == "align_session_evidence_to_jd":
            return tool_align_session_evidence_to_jd(
                session_id=args.get("session_id", ""),
                position_id=args.get("position_id", ""),
                jd_text=args.get("jd_text", ""),
            )
        raise KeyError(f"未知 MCP 工具：{name}")


_SERVER = CareerMcpServer()


def handle_jsonrpc(req: dict[str, Any]) -> dict[str, Any] | None:
    """处理单条 MCP JSON-RPC 2.0 请求；notification 无 id 时返回 None。"""
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}

    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SPEC}}
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        try:
            structured = _SERVER.dispatch_tool(name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(structured, ensure_ascii=False),
                        }
                    ],
                    "structuredContent": structured,
                    "isError": False,
                },
            }
        except KeyError as exc:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": str(exc)}}
        except ValueError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> int:
    """标准 stdio 行分隔 JSON-RPC 2.0 MCP 服务循环。"""
    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        try:
            req = json.loads(s)
        except json.JSONDecodeError as exc:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {exc}"}}
        else:
            resp = handle_jsonrpc(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
