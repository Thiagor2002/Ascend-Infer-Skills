# Ascend 性能寻优 Skill

给 Cursor、Codex、Claude Code 使用的同一份技能。它指导 agent 在 Ascend A5（950，当前主力）及后续机型上，用 vLLM-Ascend 服务化加 AISBench 闭环压测，找到 TPOT、TTFT 与 prefix cache 命中率约束下的**最大单卡输出吞吐**，并把寻优过程建档成可复现的表。

面向 Qwen3.5、Qwen3.8 这类 mxfp8 量化 + MTP 投机 + Decode 入图的部署。不用于精度定位，不用于设计新算子。

## 安装

保留整个目录。三个平台都读取 `SKILL.md`，Codex 另外读取 `agents/openai.yaml`。

本目录是 [Ascend Infer Skills](../README.md) 中的一份技能，仓库路径为 `ascend-performance-search/`。安装时目标文件夹名保持为 frontmatter 中的 `ascend-performance-search`。

| 平台 | 目录 |
|---|---|
| Cursor | `~/.cursor/skills/ascend-performance-search` |
| Codex | `~/.codex/skills/ascend-performance-search` |
| Claude Code | `~/.claude/skills/ascend-performance-search` |

Windows 上 `~` 是 `%USERPROFILE%`。也可以放进项目的 `.cursor/skills/`、`.codex/skills/` 或 `.claude/skills/`。不要只复制 `SKILL.md`。

## 使用

显式调用：

```text
$ascend-performance-search
```

描述里出现单卡输出寻优、并发扫描、TPOT/TTFT 门限、DP/TP 布局比较、PD 混部或 PD 分离性能摸测时，agent 也应载入本技能。

建议一次提供：

- 机型、可用卡数、容器名，以及哪些卡已被占用；
- 模型权重包路径与量化方式；
- base 服务化命令，尤其是图模式、MTP 步数、DP/TP；
- 场景清单：输入/输出长度、目标 prefix cache 命中率、TPOT 与 TTFT 上限；
- 本轮是否允许重启服务、是否允许叠加特性。

示例：

```text
使用 $ascend-performance-search 在 8 卡 A5 上摸 Qwen3.8-27B 的单卡输出。
容器 <name>，权重 <path>，base 是 mxfp8 + MTP3 + 异步调度 + Decode 入图。
场景：4k/2k 与 16k/1k 零命中，64k/1k 与 128k/1k 命中 90%，都要求 TPOT 平均 < 20ms。
先做 1 卡和 2 卡混部，每条序列给出门限内最高并发及其相邻超限点，过程全部建档。
```

## 目录

```text
ascend-performance-search/
├── SKILL.md
├── README.md
├── agents/openai.yaml
├── references/
│   ├── serving-baseline.md
│   ├── aisbench-harness.md
│   ├── concurrency-search.md
│   └── result-recording.md
└── tests/manual-tests.md
```

`references/serving-baseline.md` 是服务拉起、自检与图捕获上限核算。`references/aisbench-harness.md` 是压测端配置、前缀预热对齐与请求级有效性校验。`references/concurrency-search.md` 是粗扫加细扫的搜索协议与断崖归因。`references/result-recording.md` 是目录结构与结果表头。

PD 混部矩阵、PD 分离配平和多模态图文用例在后续提交补入。
