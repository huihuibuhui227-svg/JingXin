#!/usr/bin/env python
"""
LLM 推理引擎 — CueCoT 实验阶段三
================================
调用 GPT-4o 对每个视频执行 CoT 推理并输出 12 维评分。

支持 5 种推理方法：
  - cot_full:      完整 3-stage CoT (Observation → Inference → Assessment)
  - direct:        直接打分（无 CoT baseline）
  - generic_cot:   通用 "think step by step" CoT
  - skip_s1:       消融：跳过 Stage 1 观察
  - skip_s2:       消融：跳过 Stage 2 推理

用法：
  # 默认使用 OpenAI GPT-4o
  python llm_reasoning.py --test 20 --estimate

  # 使用 DeepSeek 或其他兼容 API
  python llm_reasoning.py --test 20 --estimate \
    --base-url https://api.deepseek.com \
    --model deepseek-chat

  # 或通过环境变量配置
  export OPENAI_API_BASE=https://api.deepseek.com
  export OPENAI_MODEL=deepseek-chat
  python llm_reasoning.py --test 20 --estimate

  # 其他模式
  python llm_reasoning.py --test 20 --method cot_full
  python llm_reasoning.py --resume
  python llm_reasoning.py --test 200 --temperatures 0.0,0.3,0.7
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from cot_prompts import (
    CoTPromptBuilder,
    build_stage2_user_message,
    build_stage3_user_message,
    TARGET_DIMENSIONS,
)

# ---------- 配置 ----------
FEATURES_DIR = Path(r"D:\jingxin\experiments\results\features")
RESULTS_FILE = FEATURES_DIR / "llm_results.jsonl"
PROGRESS_FILE = FEATURES_DIR / "llm_progress.json"
COST_ESTIMATE_FILE = FEATURES_DIR / "llm_cost_estimate.json"
ROBUSTNESS_FILE = FEATURES_DIR / "temperature_robustness.json"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-2024-08-06")
DEFAULT_BASE_URL = os.getenv("OPENAI_API_BASE", "")  # 空字符串 = 用 OpenAI 默认
DEFAULT_TEMPERATURE = 0.0
MAX_TOKENS = 6000
RATE_LIMIT_RPM = 500  # GPT-4o Tier 1
MIN_CALL_INTERVAL = 60.0 / RATE_LIMIT_RPM  # seconds between calls

ALL_METHODS = ["cot_full", "direct", "generic_cot", "skip_s1", "skip_s2"]

# 重试配置
RETRYABLE_ERRORS = (
    "rate_limit",
    "timeout",
    "server_error",
    "connection_error",
    "api_connection_error",
    "internal_server_error",
    "service_unavailable",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# JSON 解析与验证
# ═══════════════════════════════════════════════════════

def extract_json_block(text: str) -> str | None:
    """从 LLM 输出中提取 JSON 块（处理 markdown 代码块包裹）。"""
    # 尝试 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    # 尝试裸 {...} 块（取最后一个完整的）
    matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if matches:
        # 找包含最多维度的 JSON
        best = max(matches, key=lambda s: sum(1 for d in TARGET_DIMENSIONS if d in s))
        return best
    return None


def parse_scores(text: str) -> tuple[dict[str, float] | None, str]:
    """从 LLM 输出解析 12 维分数。

    返回 (scores_dict_or_None, parse_status)
    status: "json_ok" | "regex_fallback" | "parse_error"
    """
    # 主路径：JSON 解析
    json_str = extract_json_block(text)
    if json_str:
        try:
            parsed = json.loads(json_str)
            scores = {}
            for dim in TARGET_DIMENSIONS:
                if dim in parsed:
                    scores[dim] = float(parsed[dim])
            if len(scores) >= 8:  # 至少要有 8 个维度才算有效
                # 补全缺失维度为 0
                for dim in TARGET_DIMENSIONS:
                    if dim not in scores:
                        scores[dim] = 0.0
                return scores, "json_ok"
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 兜底：正则提取
    scores = {}
    for dim in TARGET_DIMENSIONS:
        pattern = rf'"{re.escape(dim)}"\s*:\s*(-?\d+\.?\d*)'
        m = re.search(pattern, text)
        if m:
            scores[dim] = float(m.group(1))

    if len(scores) >= 8:
        for dim in TARGET_DIMENSIONS:
            if dim not in scores:
                scores[dim] = 0.0
        return scores, "regex_fallback"

    return None, "parse_error"


def validate_scores(scores: dict[str, float]) -> bool:
    """检查分数是否在合理范围 [-3, 3] 且包含全部 12 维。"""
    if set(scores.keys()) != set(TARGET_DIMENSIONS):
        return False
    return all(-3.0 <= v <= 3.0 for v in scores.values())


# ═══════════════════════════════════════════════════════
# LLM 推理器
# ═══════════════════════════════════════════════════════

class LLMReasoner:
    """LLM 推理引擎，封装 API 调用、重试、解析。

    支持 OpenAI 和任何 OpenAI 兼容的 API（DeepSeek, vLLM 等）。
    """

    def __init__(self, model: str = "", base_url: str = "", temperature: float = DEFAULT_TEMPERATURE, workers: int = 1):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 未设置。请设置环境变量：\n"
                "  export OPENAI_API_KEY=sk-...\n"
                "或在 .env 文件中添加 OPENAI_API_KEY=sk-..."
            )

        model = model or DEFAULT_MODEL
        base_url = base_url or DEFAULT_BASE_URL
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._workers = workers
        self.builder = CoTPromptBuilder()
        self._last_call_time = 0.0
        self._total_tokens = {"input": 0, "output": 0}

    def _rate_limit(self):
        """简单的速率限制：确保两次调用间隔 >= MIN_CALL_INTERVAL。"""
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_api(self, messages: list[dict], temperature: float | None = None) -> dict:
        """单次 API 调用，带重试。返回 {"content": str, "input_tokens": int, "output_tokens": int}。"""
        self._rate_limit()
        temp = temperature if temperature is not None else self.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=MAX_TOKENS,
                response_format={"type": "text"},
            )
        except Exception as e:
            err_str = str(e).lower()
            if any(r in err_str for r in RETRYABLE_ERRORS):
                log(f"  [RETRY] {type(e).__name__}: {e}")
                raise  # tenacity 会重试
            raise  # 非可重试错误也抛出，但会被外层捕获

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # 空响应检测（某些 API 偶发空返回，应触发重试）
        finish_reason = response.choices[0].finish_reason
        if not content.strip() and finish_reason != "stop":
            raise RuntimeError(f"Empty response from API (finish_reason={finish_reason})")

        self._total_tokens["input"] += input_tokens
        self._total_tokens["output"] += output_tokens

        return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens}

    # ---------- 5 种推理方法 ----------

    def run_cot_full(self, features: dict[str, float], temperature: float | None = None) -> dict:
        """完整 3-stage CoT。返回 scores + reasoning_chain + tokens。"""
        stages = self.builder.build_full_cot(features)
        result = {"method": "cot_full", "temperature": temperature or self.temperature, "stages": {}}
        total_tokens = {"input": 0, "output": 0}

        # Stage 1: Observation
        try:
            resp1 = self._call_api(stages[0], temperature)
            result["stages"]["stage1_output"] = resp1["content"]
            total_tokens["input"] += resp1["input_tokens"]
            total_tokens["output"] += resp1["output_tokens"]
        except Exception as e:
            result["stages"]["stage1_error"] = str(e)
            result["parse_status"] = "stage1_failed"
            return result

        # Stage 2: Inference (uses Stage 1 output)
        s2_messages = stages[1] + build_stage2_user_message(resp1["content"])
        try:
            resp2 = self._call_api(s2_messages, temperature)
            result["stages"]["stage2_output"] = resp2["content"]
            total_tokens["input"] += resp2["input_tokens"]
            total_tokens["output"] += resp2["output_tokens"]
        except Exception as e:
            result["stages"]["stage2_error"] = str(e)
            result["parse_status"] = "stage2_failed"
            return result

        # Stage 3: Assessment (uses Stage 1 + Stage 2 output)
        s3_messages = stages[2] + build_stage3_user_message(resp1["content"], resp2["content"])
        try:
            resp3 = self._call_api(s3_messages, temperature)
            result["stages"]["stage3_output"] = resp3["content"]
            total_tokens["input"] += resp3["input_tokens"]
            total_tokens["output"] += resp3["output_tokens"]
        except Exception as e:
            result["stages"]["stage3_error"] = str(e)
            result["parse_status"] = "stage3_failed"
            return result

        # 从 Stage 3 输出解析分数
        scores, parse_status = parse_scores(resp3["content"])
        result["scores"] = scores
        result["parse_status"] = parse_status
        result["scores_valid"] = validate_scores(scores) if scores else False
        result["tokens"] = total_tokens
        return result

    def run_direct(self, features: dict[str, float], temperature: float | None = None) -> dict:
        """Direct baseline: 特征 → 直接打分。"""
        messages = self.builder.build_direct(features)
        result = {"method": "direct", "temperature": temperature or self.temperature}

        try:
            resp = self._call_api(messages, temperature)
            result["raw_output"] = resp["content"]
            scores, parse_status = parse_scores(resp["content"])
            result["scores"] = scores
            result["parse_status"] = parse_status
            result["scores_valid"] = validate_scores(scores) if scores else False
            result["tokens"] = {"input": resp["input_tokens"], "output": resp["output_tokens"]}
        except Exception as e:
            result["error"] = str(e)
            result["parse_status"] = "api_error"
        return result

    def run_generic_cot(self, features: dict[str, float], temperature: float | None = None) -> dict:
        """Generic CoT baseline: "think step by step"。"""
        messages = self.builder.build_generic_cot(features)
        result = {"method": "generic_cot", "temperature": temperature or self.temperature}

        try:
            resp = self._call_api(messages, temperature)
            result["raw_output"] = resp["content"]
            scores, parse_status = parse_scores(resp["content"])
            result["scores"] = scores
            result["parse_status"] = parse_status
            result["scores_valid"] = validate_scores(scores) if scores else False
            result["tokens"] = {"input": resp["input_tokens"], "output": resp["output_tokens"]}
        except Exception as e:
            result["error"] = str(e)
            result["parse_status"] = "api_error"
        return result

    def run_skip_s1(self, features: dict[str, float], temperature: float | None = None) -> dict:
        """Ablation: 跳过 Stage 1，特征 → 推断 + 打分。"""
        messages = self.builder.build_skip_s1(features)
        result = {"method": "skip_s1", "temperature": temperature or self.temperature}

        try:
            resp = self._call_api(messages, temperature)
            result["raw_output"] = resp["content"]
            scores, parse_status = parse_scores(resp["content"])
            result["scores"] = scores
            result["parse_status"] = parse_status
            result["scores_valid"] = validate_scores(scores) if scores else False
            result["tokens"] = {"input": resp["input_tokens"], "output": resp["output_tokens"]}
        except Exception as e:
            result["error"] = str(e)
            result["parse_status"] = "api_error"
        return result

    def run_skip_s2(self, features: dict[str, float], temperature: float | None = None) -> dict:
        """Ablation: 跳过 Stage 2，观察 → 直接打分。"""
        messages = self.builder.build_skip_s2(features)
        result = {"method": "skip_s2", "temperature": temperature or self.temperature}

        try:
            resp = self._call_api(messages, temperature)
            result["raw_output"] = resp["content"]
            scores, parse_status = parse_scores(resp["content"])
            result["scores"] = scores
            result["parse_status"] = parse_status
            result["scores_valid"] = validate_scores(scores) if scores else False
            result["tokens"] = {"input": resp["input_tokens"], "output": resp["output_tokens"]}
        except Exception as e:
            result["error"] = str(e)
            result["parse_status"] = "api_error"
        return result

    @property
    def total_tokens(self) -> dict:
        return dict(self._total_tokens)


# ═══════════════════════════════════════════════════════
# 批量运行器
# ═══════════════════════════════════════════════════════

def load_feature_vectors(jsonl_path: Path = None, limit: int = 0) -> list[dict]:
    """从 all_features.jsonl 加载特征向量。"""
    if jsonl_path is None:
        jsonl_path = FEATURES_DIR / "all_features.jsonl"
    videos = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            entry = json.loads(line.strip())
            # 提取扁平特征
            features = {}
            for k, v in entry.items():
                if k.startswith("face.") or k.startswith("gesture.") or k.startswith("voice."):
                    features[k] = float(v)
            entry["_features"] = features
            videos.append(entry)
    return videos


def load_progress() -> set[tuple[str, str, float]]:
    """加载已完成的 (video_id, method, temperature) 集合。"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            return set(tuple(x) for x in data.get("completed", []))
        except Exception:
            return set()
    return set()


def save_progress(completed: set) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "completed": [list(x) for x in completed],
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)


def estimate_cost(n_videos: int, methods: list[str], tokens_per_call: dict, model: str = "") -> dict:
    """根据样本 token 消耗估算全量成本。

    GPT-4o 定价 (2026):
      - Input:  $2.50 / 1M tokens
      - Output: $10.00 / 1M tokens
    DeepSeek / 兼容 API: 无标准定价，仅统计 token 数。
    """
    input_price = 2.50 / 1_000_000
    output_price = 10.00 / 1_000_000

    estimates = {}
    total_input_cost = 0.0
    total_output_cost = 0.0

    for method in methods:
        if method not in tokens_per_call:
            continue
        tc = tokens_per_call[method]
        method_input_cost = tc["input"] * n_videos * input_price
        method_output_cost = tc["output"] * n_videos * output_price
        method_total = method_input_cost + method_output_cost
        estimates[method] = {
            "tokens_per_video": tc,
            "cost_per_video": round(method_total / n_videos, 4) if n_videos else 0,
            "total_cost": round(method_total, 2),
        }
        total_input_cost += method_input_cost
        total_output_cost += method_output_cost

    return {
        "n_videos": n_videos,
        "n_methods": len(methods),
        "model": model,
        "input_price_per_1M": input_price * 1_000_000,
        "output_price_per_1M": output_price * 1_000_000,
        "per_method": estimates,
        "grand_total": round(total_input_cost + total_output_cost, 2),
    }


def append_result(entry: dict) -> None:
    """追加一行到 llm_results.jsonl。"""
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CueCoT LLM 推理")
    ap.add_argument("--test", type=int, default=0, help="测试前 N 个视频")
    ap.add_argument("--method", type=str, default="", help="仅运行指定方法 (cot_full/direct/generic_cot/skip_s1/skip_s2)")
    ap.add_argument("--methods", type=str, default="", help="逗号分隔的多个方法")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="LLM 温度参数")
    ap.add_argument("--temperatures", type=str, default="", help="逗号分隔的多个温度（鲁棒性测试）")
    ap.add_argument("--estimate", action="store_true", help="成本预估模式（仅统计 token 用量）")
    ap.add_argument("--resume", action="store_true", help="断点续跑")
    ap.add_argument("--video", type=str, default="", help="单视频调试")
    ap.add_argument("--model", type=str, default="", help=f"模型名称 (默认: {DEFAULT_MODEL})")
    ap.add_argument("--base-url", type=str, default="", help="API Base URL (空=OpenAI 默认)")
    ap.add_argument("--workers", type=int, default=1, help="并发线程数 (默认1, 建议8-16)")
    args = ap.parse_args()

    model = args.model or DEFAULT_MODEL
    base_url = args.base_url or DEFAULT_BASE_URL
    log("=== CueCoT LLM 推理 ===")
    log(f"API Base URL: {base_url or '(OpenAI 默认)'}")
    log(f"模型: {model}")

    # 确定方法列表
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    elif args.method:
        methods = [args.method]
    else:
        methods = list(ALL_METHODS)

    # 确定温度列表
    if args.temperatures:
        temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    else:
        temperatures = [args.temperature]

    log(f"方法: {methods}")
    log(f"温度: {temperatures}")
    log(f"输出: {RESULTS_FILE}")

    # 加载视频
    if args.video:
        # 单视频：从 feature_vector.json 加载
        fv_path = FEATURES_DIR / args.video / "feature_vector.json"
        if not fv_path.exists():
            log(f"错误: {fv_path} 不存在")
            sys.exit(1)
        with open(fv_path, encoding="utf-8") as f:
            fv = json.load(f)
        flat = {}
        for face_k, face_v in fv.get("face", {}).items():
            if isinstance(face_v, dict):
                for stat_k, stat_v in face_v.items():
                    flat[f"face.{face_k}_{stat_k}"] = float(stat_v)
        for ges_k, ges_v in fv.get("gesture", {}).items():
            if isinstance(ges_v, dict):
                for stat_k, stat_v in ges_v.items():
                    flat[f"gesture.{ges_k}_{stat_k}"] = float(stat_v)
        for voi_k, voi_v in fv.get("voice", {}).items():
            flat[f"voice.{voi_k}"] = float(voi_v)
        videos = [{"video_id": args.video, "_features": flat}]
    else:
        limit = args.test if args.test > 0 else 0
        videos = load_feature_vectors(limit=limit)
    log(f"待处理视频: {len(videos)}")

    # 加载进度
    completed = load_progress() if args.resume else set()
    if completed:
        log(f"已跳过 {len(completed)} 个已完成 (video, method, temp) 组合")

    # 初始化推理器
    try:
        reasoner = LLMReasoner(model=model, base_url=base_url, temperature=args.temperature, workers=args.workers)
    except RuntimeError as e:
        log(f"初始化失败: {e}")
        sys.exit(1)

    # 路由到模式
    if args.estimate:
        _run_estimate_mode(videos, methods, reasoner)
    elif len(temperatures) > 1:
        _run_robustness_mode(videos, methods, temperatures, reasoner, completed)
    else:
        _run_batch_mode(videos, methods, temperatures[0], reasoner, completed, args.resume)


def _run_estimate_mode(videos, methods, reasoner):
    """成本预估：运行少量视频，统计 token 消耗。"""
    sample = videos[: min(20, len(videos))]
    log(f"成本预估: {len(sample)} 个视频, {len(methods)} 种方法")

    tokens_per_method = {}
    for vid_entry in sample:
        features = vid_entry["_features"]
        vid = vid_entry["video_id"]

        for method in methods:
            func = getattr(reasoner, f"run_{method}", None)
            if func is None:
                log(f"  {method}: 未知方法，跳过")
                continue

            log(f"  [{vid}] {method}...")
            try:
                result = func(features, temperature=0.0)
            except Exception as e:
                log(f"    FAILED: {e}")
                continue

            tc = result.get("tokens", {})
            if method not in tokens_per_method:
                tokens_per_method[method] = {"input": 0, "output": 0, "calls": 0}
            tokens_per_method[method]["input"] += tc.get("input", 0)
            tokens_per_method[method]["output"] += tc.get("output", 0)
            tokens_per_method[method]["calls"] += 1
            log(f"    tokens: in={tc.get('input', 0)} out={tc.get('output', 0)} status={result.get('parse_status', '?')}")

    # 计算平均
    avg_tokens = {}
    for method, tc in tokens_per_method.items():
        n = tc["calls"]
        avg_tokens[method] = {
            "input": round(tc["input"] / n) if n > 0 else 0,
            "output": round(tc["output"] / n) if n > 0 else 0,
        }

    # 全量成本估算
    n_total = len(videos)
    estimate = estimate_cost(n_total, list(avg_tokens.keys()), avg_tokens, reasoner.model)

    log("\n=== 成本估算 ===")
    log(f"模型: {reasoner.model}")
    log(f"视频数: {n_total}")
    log(f"{'方法':<20s} {'输入tokens/视频':>15s} {'输出tokens/视频':>15s} {'成本/视频':>10s} {'总成本':>10s}")
    log("-" * 75)
    for method, info in estimate["per_method"].items():
        tc = info["tokens_per_video"]
        log(f"  {method:<18s} {tc['input']:>12,d} {tc['output']:>12,d} "
            f"${info['cost_per_video']:>8.4f} ${info['total_cost']:>8.2f}")
    log(f"{'总计':>66s} ${estimate['grand_total']:>8.2f}")

    # 保存
    with open(COST_ESTIMATE_FILE, "w", encoding="utf-8") as f:
        json.dump(estimate, f, ensure_ascii=False, indent=2)
    log(f"\n详细估算已保存到: {COST_ESTIMATE_FILE}")


def _run_batch_mode(videos, methods, temperature, reasoner, completed, resume):
    """批量运行模式，支持多线程并发。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    workers = reasoner._workers
    log(f"批量运行: {len(videos)} 视频 x {len(methods)} 方法, {workers} 并发, temperature={temperature}")

    t0 = time.time()
    processed = [0]
    failed = [0]
    parse_errors = [0]
    lock = threading.Lock()

    # 构建任务列表
    tasks = []
    for vid_entry in videos:
        features = vid_entry["_features"]
        vid = vid_entry["video_id"]
        for method in methods:
            key = (vid, method, temperature)
            if key in completed:
                continue
            tasks.append((vid, features, method, temperature))

    total_tasks = len(tasks)
    log(f"待处理任务: {total_tasks}")

    def process_one(vid, features, method, temp):
        """单个 (video, method) 任务 — 在线程中执行。"""
        # 每个线程独立创建 LLMReasoner（OpenAI client 是线程安全的，但分开更干净）
        r = LLMReasoner(model=reasoner.model, base_url=reasoner.base_url, temperature=temp)
        func = getattr(r, f"run_{method}", None)
        if func is None:
            return None

        try:
            result = func(features, temp)
        except Exception as e:
            with lock:
                failed[0] += 1
            return {"video_id": vid, "method": method, "temperature": temp,
                    "scores": None, "parse_status": "api_error", "error": str(e),
                    "tokens": {"input": 0, "output": 0}}

        entry = {
            "video_id": vid,
            "method": method,
            "temperature": temp,
            "scores": result.get("scores"),
            "parse_status": result.get("parse_status"),
            "scores_valid": result.get("scores_valid"),
            "tokens": result.get("tokens"),
        }
        if "stages" in result:
            entry["stages"] = result["stages"]
        if "raw_output" in result:
            entry["raw_output"] = result["raw_output"]
        if "error" in result:
            entry["error"] = result["error"]

        with lock:
            append_result(entry)
            completed.add((vid, method, temp))
            processed[0] += 1
            if result.get("parse_status") == "parse_error":
                parse_errors[0] += 1

        return entry

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, v, f, m, t): (v, m) for v, f, m, t in tasks}
        for future in as_completed(futures):
            vid, method = futures[future]
            try:
                future.result()
            except Exception as e:
                with lock:
                    failed[0] += 1
                log(f"  [{vid}] {method} THREAD CRASH: {e}")

            # 每完成 50 个任务或每 2% 报进度
            with lock:
                p = processed[0]
            report_every = min(50, max(1, total_tasks // 50))
            if p % report_every == 0:
                elapsed = time.time() - t0
                rate = p / elapsed * 60 if elapsed > 0 else 0
                log(f"  进度 {p}/{total_tasks} ({rate:.1f} tasks/min) parse_err={parse_errors[0]}")
                if resume:
                    save_progress(completed)

    elapsed = time.time() - t0
    log(f"=== 完成 === 处理: {processed[0]} 失败: {failed[0]} 解析错误: {parse_errors[0]} "
        f"耗时: {elapsed/60:.1f} min")
    save_progress(completed)


def _run_robustness_mode(videos, methods, temperatures, reasoner, completed):
    """温度鲁棒性测试：每个视频用多个温度运行，计算分数方差。"""
    log(f"鲁棒性测试: {len(videos)} 视频 × {len(methods)} 方法 × {len(temperatures)} 温度")

    robustness_results = []
    t0 = time.time()

    for i, vid_entry in enumerate(videos):
        features = vid_entry["_features"]
        vid = vid_entry["video_id"]
        vid_result = {"video_id": vid, "methods": {}}

        for method in methods:
            method_scores = {}
            for temp in temperatures:
                key = (vid, method, temp)
                if key in completed:
                    continue

                func = getattr(reasoner, f"run_{method}", None)
                if func is None:
                    continue

                try:
                    result = func(features, temp)
                except Exception as e:
                    log(f"  [{vid}] {method} T={temp} CRASH: {e}")
                    continue

                scores = result.get("scores", {})
                if scores:
                    method_scores[temp] = scores

                # 保存单个结果
                entry = {
                    "video_id": vid,
                    "method": method,
                    "temperature": temp,
                    "scores": scores,
                    "parse_status": result.get("parse_status"),
                    "tokens": result.get("tokens"),
                }
                append_result(entry)
                completed.add(key)

            if len(method_scores) >= 2:
                # 计算跨温度的分数方差
                dim_variances = {}
                for dim in TARGET_DIMENSIONS:
                    vals = [ms[dim] for ms in method_scores.values() if dim in ms]
                    if len(vals) >= 2:
                        import numpy as np
                        dim_variances[dim] = {
                            "mean": float(np.mean(vals)),
                            "std": float(np.std(vals)),
                            "values": {str(t): ms[dim] for t, ms in method_scores.items() if dim in ms},
                        }
                vid_result["methods"][method] = {
                    "n_temperatures": len(method_scores),
                    "dimension_variances": dim_variances,
                }

        robustness_results.append(vid_result)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            log(f"  鲁棒性进度 {i + 1}/{len(videos)} ({elapsed:.0f}s)")

    # 汇总统计
    import numpy as np
    summary = {"n_videos": len(videos), "n_methods": len(methods), "n_temperatures": len(temperatures)}
    for method in methods:
        method_dims = {dim: [] for dim in TARGET_DIMENSIONS}
        for vr in robustness_results:
            if method in vr["methods"]:
                for dim, info in vr["methods"][method].get("dimension_variances", {}).items():
                    method_dims[dim].append(info["std"])

        method_summary = {}
        for dim, stds in method_dims.items():
            if stds:
                method_summary[dim] = {
                    "mean_std_across_videos": float(np.mean(stds)),
                    "max_std": float(np.max(stds)),
                }
        summary[method] = method_summary

    with open(ROBUSTNESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": robustness_results}, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log(f"鲁棒性测试完成: {elapsed/60:.1f} min")
    log(f"结果保存到: {ROBUSTNESS_FILE}")
    save_progress(completed)


if __name__ == "__main__":
    main()
