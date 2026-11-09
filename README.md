# Ascend Infer Skills

收录用于 Ascend 硬件推理的 Agent Skills。每份技能单独放在一个子目录里，可以分别安装到 Cursor、Codex 和 Claude Code。仓库根目录只做索引，不是某一份技能的安装根。

三端都读取技能目录中的 `SKILL.md`。Codex 另外读取 `agents/openai.yaml`。目录遵循开放的 Agent Skills 结构：`SKILL.md` 放触发条件和主流程，按需细节放在一级 `references/`。参考：[OpenAI Skills 指南](https://developers.openai.com/api/docs/guides/tools-skills)、[Cursor Skills 文档](https://cursor.com/docs/skills)、[Claude Agent Skills 概览](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)。

## 技能

| 源目录 | 安装名 | 做什么 |
|---|---|---|
| [`ascend-310p-performance-skills/`](ascend-310p-performance-skills/) | `ascend-310p-performance` | 看护 Ascend 310P 上 vLLM-Ascend 的推理性能，并修复明显拖慢性能的代码。对照无 Triton、FRACTAL_NZ、paged attention 和 event-id 等硬件约束。面向 Qwen3、Qwen3-VL、Qwen3.5、Qwen3.8，以及 Gemma 的 RMSNorm 差异。不用于设计新优化，也不用于纯精度定位。 |

安装名来自该技能 `SKILL.md` frontmatter 的 `name`，与仓库里的源目录名可以不同。触发条件、调用示例和边界写在对应目录的 `README.md` 与 `SKILL.md`。

性能技能把纯精度回归交给 `ascend-310p-accuracy-troubleshooting`。那份技能尚未收入本仓库；收入时按同样的子目录方式放置，并在上表补一行。

## 安装

复制某一份技能的整个目录，不要复制仓库根。安装后的文件夹名使用安装名。

| 平台 | 目录 |
|---|---|
| Cursor | `~/.cursor/skills/<安装名>` |
| Codex | `~/.codex/skills/<安装名>` |
| Claude Code | `~/.claude/skills/<安装名>` |

Windows 上 `~` 是 `%USERPROFILE%`。也可以放进项目的 `.cursor/skills/`、`.codex/skills/` 或 `.claude/skills/`。保留 `SKILL.md`、`references/` 和 `agents/`，不要只复制 `SKILL.md`。复制后重新打开会话。

当前性能技能复制到 `ascend-310p-performance`，显式调用 `$ascend-310p-performance`。

## 目录

```text
Ascend-Infer-Skills/
├── README.md
└── ascend-310p-performance-skills/
    ├── SKILL.md
    ├── README.md
    ├── agents/
    │   └── openai.yaml
    ├── references/
    │   ├── hardware-constraints.md
    │   ├── aisbench-verification.md
    │   ├── operator-substitution.md
    │   └── graph-mtp-models.md
    └── appendix/
        ├── optimization-map.md
        └── profiler-analysis.md
```

## 新增技能

在仓库根下新增一个独立目录，自带 `SKILL.md`、`README.md`、`agents/openai.yaml` 和 `references/`。然后在上面的技能表和目录树里补上该技能。技能之间只通过安装名互相交接，不把另一份技能的流程复制进 `SKILL.md`。
