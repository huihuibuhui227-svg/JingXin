"""题库：面试 / 科研两个模块，**每个模块可以存多套题库**，可上传替换，参考答案可选。

## 为什么是"多套"而不是"一份"

2026-10-26 使用者原话：「题库为什么是按题目修改？这样不就只能用一份题库了吗」。
一份题库被原地改，等于每次换题都把上一份覆盖掉；而真实用法是**按岗位/批次准备
多套**，录之前选一套。所以存储改成：

```json
{
  "interview": {
    "active": "b1",
    "banks": [
      {"id": "b1", "name": "2026 秋招", "updated_at": "…",
       "questions": [{"question": "…", "reference_answer": "…"}]}
    ]
  },
  "research": { "...": "同上" }
}
```

* `active` = 下一场评估实际用的那一套（`null` = 内置默认）；
* `banks` = 这个模块下所有自定义题库；
* 没有自定义题库时回退到内置默认（开箱即用）。

## 消费方

* 语音服务在新开一场时读 `active`（`get_question_texts`）；
* 总控面板提供查看 / 新建 / 导入 / 编辑 / 删除 / 设为当前。

`reference_answer` 现在**只存不用**：它给后续的参考答案比对/评分留的输入，不参与
本场任何指标计算。不要因为它存在就假装它已经影响了结果。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

#: 模块 → 中文名。键就是接口路径里的 `<module>`。
MODULES: dict[str, str] = {"interview": "面试", "research": "科研"}

MAX_QUESTIONS = 50
MAX_QUESTION_CHARS = 500
MAX_REFERENCE_CHARS = 2000
MAX_NAME_CHARS = 80

BANK_PATH = Path(__file__).resolve().parent / "data" / "question_bank.json"

_LOCK = threading.Lock()

#: 内置默认题库 —— 与 `voice_interaction/pipeline/assessment_pipeline.py` 里原来写死
#: 的那两张表**逐字一致**。改这里等于改"没上传过题库的机器会问什么"。
DEFAULT_QUESTIONS: dict[str, list[str]] = {
    "interview": [
        "请简单介绍一下你自己，包括教育背景和研究兴趣。",
        "你为什么想从事科研工作？",
        "描述一次你解决复杂问题的经历。",
        "你在团队合作中通常扮演什么角色？",
        "你如何应对科研中的失败或挫折？",
        "你最近读过哪些与你研究方向相关的论文？",
        "你未来五年的职业规划是什么？",
        "你有什么问题想问我们吗？",
    ],
    "research": [
        "请描述一个你深入研究过的技术、科学或学术问题，你是如何解决的？",
        "当你遇到无法立即解决的难题时，通常会采取哪些步骤？",
        "你如何判断一个研究课题是否值得深入探索？",
        "请分享一次你通过批判性思维发现他人研究中漏洞的经历。",
        "在科研中，你如何平衡创新与可行性？",
        "你通常如何验证你的假设或实验结果的可靠性？",
        "描述一次你从失败实验中学到重要经验的经历。",
        "你认为优秀的科研工作者最重要的三个特质是什么？为什么？",
    ],
}


def _check_module(module: str) -> str:
    if module not in MODULES:
        raise ValueError(f"未知模块 {module!r}（只允许 {', '.join(MODULES)}）")
    return module


def _read_all() -> dict:
    """读整份题库文件。**文件不存在 = 空**(第一次上传前就是这种状态)。"""
    try:
        raw = BANK_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"题库文件损坏({BANK_PATH}):{exc}") from exc
    return data if isinstance(data, dict) else {}


def _write_all(data: dict) -> None:
    """临时文件 + `os.replace`(与仓里其他落盘同一讲究:不留半个文件)。"""
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".question_bank.", suffix=".tmp",
                               dir=str(BANK_PATH.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, BANK_PATH)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _clean_items(items: Any) -> list[dict]:
    """把上传的形状收紧成 `[{"question": ..., "reference_answer": ...}]`。"""
    if not isinstance(items, list) or not items:
        raise ValueError("题库不能为空 —— 至少要有一题")
    if len(items) > MAX_QUESTIONS:
        raise ValueError(f"题目太多:{len(items)} 题,上限 {MAX_QUESTIONS}")
    out: list[dict] = []
    for i, raw in enumerate(items, 1):
        if isinstance(raw, str):
            q, ref = raw, ""
        elif isinstance(raw, dict):
            q = raw.get("question", "")
            ref = raw.get("reference_answer", "") or ""
        else:
            raise ValueError(f"第 {i} 题格式不对:要字符串或 {{question, reference_answer}}")
        q = str(q).strip()
        ref = str(ref).strip()
        if not q:
            raise ValueError(f"第 {i} 题题干是空的")
        if len(q) > MAX_QUESTION_CHARS:
            raise ValueError(f"第 {i} 题题干 {len(q)} 字,超过上限 {MAX_QUESTION_CHARS}")
        if len(ref) > MAX_REFERENCE_CHARS:
            raise ValueError(f"第 {i} 题参考答案 {len(ref)} 字,超过上限 {MAX_REFERENCE_CHARS}")
        out.append({"question": q, "reference_answer": ref})
    return out


def _clean_name(name: Any, existing: list[dict]) -> str:
    n = str(name or "").strip()
    if not n:
        n = f"题库 {len(existing) + 1}"
    if len(n) > MAX_NAME_CHARS:
        raise ValueError(f"题库名 {len(n)} 字,超过上限 {MAX_NAME_CHARS}")
    return n


def _state(module: str) -> dict:
    """模块状态。同时**兼容旧格式**(单题库:questions 直接挂在模块下)。"""
    module = _check_module(module)
    data = _read_all()
    rec = data.get(module)
    if isinstance(rec, dict) and isinstance(rec.get("banks"), list):
        banks = []
        for b in rec["banks"]:
            if not isinstance(b, dict):
                continue
            try:
                questions = _clean_items(b.get("questions"))
            except ValueError:
                continue
            banks.append({
                "id": str(b.get("id") or uuid.uuid4().hex[:8]),
                "name": str(b.get("name") or "题库").strip() or "题库",
                "updated_at": b.get("updated_at"),
                "questions": questions,
            })
        active = rec.get("active")
        if active not in {b["id"] for b in banks}:
            active = None
        return {"active": active, "banks": banks}
    # 旧格式迁移:一份题库 → 一套名为「导入的题库」的 bank
    if isinstance(rec, dict) and isinstance(rec.get("questions"), list):
        try:
            questions = _clean_items(rec["questions"])
        except ValueError:
            return {"active": None, "banks": []}
        bid = "legacy"
        return {"active": bid, "banks": [{
            "id": bid, "name": "导入的题库",
            "updated_at": rec.get("updated_at"), "questions": questions,
        }]}
    return {"active": None, "banks": []}


def load(module: str) -> dict:
    """→ 当前模块的全部题库 + 当前生效的那一套。"""
    module = _check_module(module)
    state = _state(module)
    active_id = state["active"]
    active_bank = next((b for b in state["banks"] if b["id"] == active_id), None)
    if active_bank is None:
        active = {
            "id": None,
            "name": "内置默认题库",
            "updated_at": None,
            "questions": [{"question": q, "reference_answer": ""}
                          for q in DEFAULT_QUESTIONS[module]],
        }
        source = "default"
    else:
        active = active_bank
        source = "custom"
    return {
        "module": module,
        "module_label": MODULES[module],
        "active_id": active_bank["id"] if active_bank else None,
        "source": source,
        "banks": state["banks"],
        "active": active,
        "defaults": [{"question": q, "reference_answer": ""}
                     for q in DEFAULT_QUESTIONS[module]],
    }


def create_bank(module: str, name: Any, items: Any) -> dict:
    """新建一套题库。**不动**其它套（这就是"多套"的意义）。"""
    module = _check_module(module)
    questions = _clean_items(items)
    with _LOCK:
        data = _read_all()
        state = _state(module)
        banks = state["banks"]
        bank = {
            "id": uuid.uuid4().hex[:8],
            "name": _clean_name(name, banks),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "questions": questions,
        }
        banks.append(bank)
        active = state["active"] or bank["id"]      # 第一套自动设为当前
        data[module] = {"active": active, "banks": banks}
        _write_all(data)
    return load(module)


def update_bank(module: str, bank_id: str, *, name: Any = None,
                items: Any = None) -> dict:
    """编辑某**一套**。改名/改题都只影响这一套。"""
    module = _check_module(module)
    with _LOCK:
        data = _read_all()
        state = _state(module)
        banks = state["banks"]
        hit = next((b for b in banks if b["id"] == bank_id), None)
        if hit is None:
            raise ValueError(f"题库不存在:{bank_id}")
        if items is not None:
            hit["questions"] = _clean_items(items)
        if name is not None:
            hit["name"] = _clean_name(name, banks)
        hit["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        data[module] = {"active": state["active"], "banks": banks}
        _write_all(data)
    return load(module)


def delete_bank(module: str, bank_id: str) -> dict:
    module = _check_module(module)
    with _LOCK:
        data = _read_all()
        state = _state(module)
        banks = [b for b in state["banks"] if b["id"] != bank_id]
        if len(banks) == len(state["banks"]):
            raise ValueError(f"题库不存在:{bank_id}")
        active = state["active"]
        if active == bank_id:
            active = None                            # 删掉当前那套 → 回退内置默认
        data[module] = {"active": active, "banks": banks}
        _write_all(data)
    return load(module)


def activate(module: str, bank_id: Any) -> dict:
    """设为当前。`None` / `"default"` = 用内置默认题库。"""
    module = _check_module(module)
    with _LOCK:
        data = _read_all()
        state = _state(module)
        if bank_id in (None, "", "default"):
            target = None
        else:
            if bank_id not in {b["id"] for b in state["banks"]}:
                raise ValueError(f"题库不存在:{bank_id}")
            target = bank_id
        data[module] = {"active": target, "banks": state["banks"]}
        _write_all(data)
    return load(module)


def get_question_texts(module: str) -> list[str]:
    """当前生效题库的题面 —— 语音服务在用。"""
    return [q["question"] for q in load(module)["active"]["questions"]]


def get_reference_answers(module: str) -> list[str]:
    """当前生效题库的参考答案(可空串)。目前没有消费方。"""
    return [q.get("reference_answer", "") for q in load(module)["active"]["questions"]]
