---
name: ascend-310p-performance
description: >-
  看护 Ascend 310P 上 vLLM-Ascend 的推理性能，并修复明显拖慢性能的代码问题。对照无 Triton、FRACTAL_NZ、paged attention 与 event-id 等硬件约束。面向 Qwen3/3.5/3.8 与 Gemma。在讨论 310P 性能回退、TTFT、TPOT、图模式、MTP 或热路径上的同步、拷贝、格式时使用。不用于设计新优化，不用于纯精度定位，不把 profiler 当作默认入口。
---

# Ascend 310P 性能看护

目标是确认 310P 上现有性能没有被破坏，并修掉明显违反硬件约束、因而变慢或无法运行的代码。典型问题是热路径上的 Triton、不该出现的 ND/NZ 转换、非法的 page 或 block、多余的 event 与同步、pageable 拷贝。

本技能不设计新的融合算子，也不从 profiler 出发规划优化。代码路径、服务指标和已有 trace 是不同层证据。PR 已合入不等于在当前 CANN/PTA、模型 shape 和配置上仍然成立。

本技能不绑定特定 agent。用当前环境实际具备的文件读取和命令执行能力。没有 310P 时只做静态检查，并标明未实测。

## 按任务读取

- 判断路径是否违反 310P 硬件约束：读 [references/hardware-constraints.md](references/hardware-constraints.md)。
- 图模式、MTP、Qwen/Gemma 路径差异：读 [references/graph-mtp-models.md](references/graph-mtp-models.md)。
- 设计修复前后的分项性能实验：读 [references/aisbench-verification.md](references/aisbench-verification.md)。
- 热路径漏接了仓库里已经落地的 310P kernel：读 [references/operator-substitution.md](references/operator-substitution.md)。只接已有调用，不搜索新算子。

一次只读当前问题需要的资料。

附录默认不读，也不要按附录设计新优化：

- 用户明确要求对照某次已合入优化有没有回退：读 [appendix/optimization-map.md](appendix/optimization-map.md)。
- 用户已经给出 trace，或明确要求解读某份 profiler：读 [appendix/profiler-analysis.md](appendix/profiler-analysis.md)。没有 trace 时不要编造 op 时间，也不要为此新开采集。

## 先分型

1. **看护**：以前更快，或 MRV1 快于 MRV2，或一次改动可能打破 Triton、NZ、page、event-id 约束。先对照 hardware-constraints，再做单变量复测。
2. **明显代码问题**：热路径上能直接指出错误调用、错误格式、非法 page、多余同步或 host 拷贝。只改这一处。
3. **服务配置**：图、MTP、并发、batch 或 `max-model-len` 踩到 310P 限制。一次只改一个变量，用 aisbench-verification 的层级。启动失败、OOM、capture 失败记 blocked，不写成性能数字。

精度回归交给 `ascend-310p-accuracy-troubleshooting`。用户同时要性能和精度时，修复仍要带本技能的精度护栏，但不能把性能实验当成精度根因定位。

## 工作流

### 1. 固定口径

能从命令、日志或源码得到的字段自己查，不重复问。至少记下：

- vLLM 与 vLLM-Ascend 的路径、branch、commit；CANN、PTA、驱动；
- 模型、量化、TP、物理卡、`ASCEND_RT_VISIBLE_DEVICES`；
- 图模式、`cudagraph_capture_sizes`、MTP 步数、prefix cache、max length / max seqs / batched tokens；
- 主指标（TTFT、TPOT、output throughput、request throughput 之一）和对比基线；
- 用户明确不让改的开关。已有 trace 只作为附加证据，不主动要求补采。

当前树的 310P MRV2 不支持 MLA、sleep、DP、PP、prefill/decode context parallel、EP、非 MTP 投机和 KV transfer。LoRA、prefix caching、Qwen3-VL/MRoPE 与 Qwen3.5 hybrid 已在当前校验入口放行，不能沿用旧限制表误报。配置踩中真实限制时，先报告启动失败原因。

### 2. 建立可比较运行

沿用用户已有服务则先核对运行时 import 路径和 commit。新起服务时从确认过的 vLLM-Ascend 源码目录启动，记录 `pwd` 和 HEAD。每个候选重启进程，不复用 KV cache 或 prefix cache。不抢占已有 NPU 任务。

`enable_npugraph_ex=false` 保持为 310P 约束，除非用户给出该版本已经验证过的其他设置。

### 3. 取证

按成本递增，在前一层已经能分开两个候选时停：

1. 读代码，标出违反 hardware-constraints 的调用：Triton、逐步 format cast、非法 page/block、capture 内搬元数据、多余 event 或全流同步、pageable 拷贝、`.item()` / `.tolist()`。
2. 单请求、temperature=0、eager、MTP 关，拉长输出，看稳态 TPOT。
3. 只打开可疑开关（图、MTP、某个格式或同步）再测。
4. 目标并发的 ais_bench。只在前面几层已经解释得通，且用户要发布口径时使用。

重复至少 3 次，报中位数和极差。warmup 不计入。OOM、捕获失败、启动失败记 blocked，不记成变慢。

### 4. 改动

用户要求修改时才改代码。一个补丁只动一个明确的代码问题。方案里写明违反的硬件约束、代码位置、改成哪条已有路径、等价条件和精度护栏。

合入前的最低验证：数值或固定 prompt 的 token 对比，加上对应性能层级的前后中位数。图、MTP 或自定义算子还要跑覆盖该路径的单测。用户要求全量数据集时再跑全量，不把小样本说成榜单恢复。

## 判断原则

- 310P 不能执行 Triton。热路径上的 Triton 是缺陷，不是待移植的优化。
- 支持的 matmul 权重和 attention K/V 用 FRACTAL_NZ，转换放在加载阶段。FP32、k=1、n=1 和 MoE gating logits 不要转 NZ。Mamba 状态保持 ND。
- paged attention 不满足 head、block 或 page 对齐时先修配置。capture 内 pageable H2D 是非法的。
- event-id 有限。TP>1 时能捕获的图数量随模型深度下降。不要为了保险加 stream、event 或全流 `synchronize()`。
- 记账留在 CPU，计算留在 NPU。pageable tensor 上的 `non_blocking=True` 不是真异步。
- 小向量用一次 bulk H2D，不用逐 index `index_copy_`。二维拷贝保持连续，避免 310P 走 strided copy。
- 开图后仍看到完整 eager op 链，就是没有 replay。先查 capture size 和 attention state。
- MTP 报告必须带平均接受长度。接受率下降时先查正确性。
- `ignore_eos=True` 的性能结果不能当作精度证据。

## 交付

- 口径：commit、模型、量化、图、MTP、并发、主指标；
- 问题：违反了哪条硬件约束，或哪一段热路径明显在做 host 同步、拷贝、格式转换；
- 证据级别：静态代码检查、单请求实测、已有 profiler、目标并发服务实测分别标注。没有的层写未测；
- 对比：单变量两侧的中位数和极差，以及精度护栏的结果；
- 结论对应到哪一层实验，不把单请求 TPOT 说成并发吞吐；
- 未改或未验证的风险，以及下一层最有区分度的实验。

## 边界

- 不把附录中的优化地图或 profiler 笔记当成待实现清单。
- 不新增 Triton kernel，不把其他设备的 Triton 或 ND 主路径原样搬到 310P。
- 不发明 `torch_npu` API。仓库和当前环境都找不到的调用保持原实现。
- 不升级、降级或卸载驱动、固件、CANN、PTA。只有证据指向该组件且用户明确同意时才动。
- 不终止其他任务，不批量删除数据集、trace 或构建目录。
- 不把 service profiler 里缺失的 v1 span 解释成 MRV2 host 侧没有开销。
