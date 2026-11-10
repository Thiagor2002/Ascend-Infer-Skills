# Ascend 310P 性能 Skill

给 Cursor、Codex、Claude Code 使用的同一份技能。它指导 agent 看护 Ascend 310P 上 vLLM-Ascend 的推理性能，并修复明显拖慢性能的代码：Triton、错误的 FRACTAL_NZ、非法的 paged attention、多余的 event 或同步、热路径上的 host 拷贝。

面向 Qwen3、Qwen3-VL、Qwen3.5、Qwen3.8，以及 Gemma 的 RMSNorm 差异。不用于设计新优化，也不用于纯精度定位。

## 安装

保留整个目录。三个平台都读取 `SKILL.md`，Codex 另外读取 `agents/openai.yaml`。

本目录是 [Ascend Infer Skills](../README.md) 中的一份技能，仓库路径为 `ascend-310p-performance-skills/`。安装时目标文件夹名保持为 frontmatter 中的 `ascend-310p-performance`。

复制到对应技能目录后重新打开会话：

| 平台 | 目录 |
|---|---|
| Cursor | `~/.cursor/skills/ascend-310p-performance` |
| Codex | `~/.codex/skills/ascend-310p-performance` |
| Claude Code | `~/.claude/skills/ascend-310p-performance` |

Windows 上 `~` 是 `%USERPROFILE%`。也可以放进项目的 `.cursor/skills/`、`.codex/skills/` 或 `.claude/skills/`。不要只复制 `SKILL.md`。

目录遵循开放的 Agent Skills 结构：`SKILL.md` 只放触发条件和主流程，执行时需要的细节放在一级 `references/`。`appendix/` 是附加资料，默认不加载。参考：[OpenAI Skills 指南](https://developers.openai.com/api/docs/guides/tools-skills)、[Cursor Skills 文档](https://cursor.com/docs/skills)、[Claude Agent Skills 概览](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) 与 [Claude 最佳实践](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)。

## 使用

显式调用：

```text
$ascend-310p-performance
```

描述里包含 310P 性能回退、TTFT、TPOT、图模式、MTP，或热路径上的 Triton、格式、同步、拷贝时，agent 也应载入本技能。

建议一次提供：

- vLLM-Ascend 源码路径、commit、模型与量化；
- 服务命令，尤其是图模式、MTP、TP 和 batch；
- 主指标和现有基线；
- 本轮是否允许改代码、是否允许占空闲卡。

示例：

```text
使用 $ascend-310p-performance 看护 310P 上 Qwen3.5 的 MTP TPOT。
源码在 <path>，服务命令为 <command>。主指标是稳态 TPOT。
先不要改代码。若热路径出现 Triton、逐步 ND/NZ 转换、pageable 拷贝或多余 event，指出对应代码并给出单变量验证方案。
```

## 目录

```text
ascend-310p-performance/
├── SKILL.md
├── README.md
├── agents/openai.yaml
├── references/
│   ├── hardware-constraints.md
│   ├── aisbench-verification.md
│   ├── operator-substitution.md
│   └── graph-mtp-models.md
└── appendix/
    ├── optimization-map.md
    └── profiler-analysis.md
```

`references/hardware-constraints.md` 从当前 `HardwareProfile` 列出 310P 登记的能力，并写 Triton、FRACTAL_NZ、paged attention 和 event-id 限制。`references/graph-mtp-models.md` 看护 MTP 叠加 ACLGraph。`references/operator-substitution.md` 只接回漏掉的已有 kernel。`appendix/` 保留原先的优化地图和 profiler 笔记，只在用户明确要求对照历史不变量或解读已有 trace 时打开。

背景分析见 `work-docs/310p-perf/`。技能正文不依赖该目录也能执行。
