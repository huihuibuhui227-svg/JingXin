#!/usr/bin/env bash
# 装 PII 守卫成 pre-commit 钩子 —— 公开仓库不许再进真实姓名+学号。
#
# ## 为什么要有这一步
#
# 扫描器写得再对,不接进提交流程就等于没有。本仓反复栽的形态是
# 「闸装了但没接到路上」(同族:「验证跑错了对象」)。
# `tests/test_pii_guard.py::test_the_guard_is_wired_as_a_pre_commit_hook` 守这一条。
#
# ## 判据只看**新增行**
#
# `git diff --cached` 里删掉一行真实姓名是清理动作,不该被拦 ——
# 拦住它等于让人没法修。见 `jx_pii_guard.scan_patch`。
#
# ## 用法
#
#   bash tools/install_pii_hook.sh          # 装
#   bash tools/install_pii_hook.sh --check  # 只检查装没装
#   git commit --no-verify                  # 明知故犯时绕过(会被 code review 问)
#
# ⚠️ 钩子**不进 git**(`.git/hooks/` 不受版本控制)⟹ 每台机器、每个克隆
# 都要跑一次。两台工作副本(WSL + gpuserver)都要装。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 解释器按机器解析(本仓纪律:脚本里不写死 `~/miniconda3/...`)。
# 守卫本身只用标准库,但用项目那个 env 保证版本与 CI 一致。
source "$REPO/tools/jx_env.sh"
PY="$JX_PY"
HOOK="$REPO/.git/hooks/pre-commit"

if [ "${1:-}" = "--check" ]; then
    if [ -x "$HOOK" ] && grep -q jx_pii_guard "$HOOK" 2>/dev/null; then
        echo "✓ PII 钩子已装:$HOOK"
        exit 0
    fi
    echo "✗ PII 钩子没装 —— 跑一次 bash tools/install_pii_hook.sh" >&2
    exit 1
fi

if [ ! -d "$REPO/.git" ]; then
    echo "✗ $REPO 不是 git 仓库的根" >&2
    exit 2
fi

cat > "$HOOK" <<HOOKEOF
#!/usr/bin/env bash
# 自动生成,源在 tools/install_pii_hook.sh —— 别手改这个文件。
# 公开仓库的 PII 守卫:素材目录名里的真实姓名 + 学号不许进仓库。
set -euo pipefail

REPO="\$(git rev-parse --show-toplevel)"
PATCH="\$(mktemp)"
trap 'rm -f "\$PATCH"' EXIT

# --cached:只看这次要提交的内容;--diff-filter=d:跳过删除。
git -C "\$REPO" diff --cached --unified=0 > "\$PATCH"

if ! "$PY" "\$REPO/tools/jx_pii_guard.py" --patch "\$PATCH"; then
    echo "" >&2
    echo "提交被拦下:上面的目录名带着真实姓名+学号,而本仓远端是公开仓库。" >&2
    echo "改成假值(学号全 0 自动豁免):示例甲-000000001-示例学院__<sid>" >&2
    echo "确认是误报时可以 git commit --no-verify 绕过。" >&2
    exit 1
fi
HOOKEOF

chmod +x "$HOOK"
echo "✓ 已装:$HOOK"
echo "  自检:bash tools/install_pii_hook.sh --check"
