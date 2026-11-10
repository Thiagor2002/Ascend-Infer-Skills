# 图模式、MTP 与模型差异

实验涉及 ACLGraph、MTP，或要判断 Qwen / Gemma 能不能在 310P 上测性能时读取。本文用来看护现有 MTP 叠加图模式有没有退回 eager、有没有抓错 attention，不用来设计新的图或新的 draft kernel。

服务命令和 ais_bench 层级见 [aisbench-verification.md](aisbench-verification.md)。event-id、paged attention 和 `enable_npugraph_ex` 见 [hardware-constraints.md](hardware-constraints.md)。精度根因交给 `ascend-310p-accuracy-troubleshooting`；这里只标出 MTP 图路径上会直接表现为慢或接受率异常的代码点。

符号以当前 `vllm_ascend/_310p/worker/v2/` 为准。行号会变。

## MTP 叠加 ACLGraph 支持到哪

310P MRV2 的投机解码只接受 `method="mtp"`。ngram 和其他投机方法在 `_validate_config` 和 sampler 构造处直接 `NotImplementedError`。不要为了测性能改掉这个检查。

图模式对外仍叫 cudagraph。MTP 叠加图时用 `FULL_DECODE_ONLY`：prefill、chunked prefill 和 prefix-cache hit 保持 eager，只有均匀的 decode / MTP verify 才 replay。`enable_npugraph_ex=false` 保持，除非该版本另有验证记录。310P 没有 FIA/PA `graph_task`。

硬件限制：

- capture size 的单位是 token，不是请求数。纯 decode 的 batch 4 是 4；MTP K=1 的 batch 4 是 8，因为 `q_len = 1+K`。尺寸对不上就整步 eager。
- TP=1 没有文档中的 event-id 捕获张数限制。TP>1 时能捕获的图数量受 event-id 和模型深度限制；文档例子是 Qwen3-32B 可以捕获并 replay 2 张图。MTP 还要额外捕获 draft 图，和 target 图共用这份预算。捕获失败记 blocked，不记成 TPOT 变慢。
- capture 内不能做 pageable H2D。`seq_lens` 和 draft 的 `slot_mapping` 必须在进图前已经在 device 上，或在图外由 host 算完再写入稳定 buffer。

K=1 与 K>1 不是同一张 draft 图。不要用 K=1 的 TPOT 或接受率代表 K>1。

## 代码结构

两条图分开捕获，不要把 draft replay 说成 target 图已经生效。

| 角色 | 类 | 文件 | 图里有什么 |
|---|---|---|---|
| Target verify | `ModelAclGraphManager310` | `_310p/worker/v2/aclgraph.py` | 均匀 MTP batch，`q_len = 1+K`，attention 标成 `SpecDecoding`（splitfuse），不是 `DecodeOnly`（PA） |
| 是否允许 replay | `NPUModelRunner310V2` | `_310p/worker/v2/model_runner.py` | prefill、prefix-cache hit、非均匀 spec batch 强制 eager；均匀 verify 从 `ChunkedPrefill` 提升回 `SpecDecoding` |
| Draft | `AscendMTPSpeculator310` + `AutoRegressiveAclGraphManager310` | `_310p/worker/v2/spec_decode/mtp_speculator.py`、`spec_decode/aclgraph.py` | K=1 的 draft-prefill FULL，以及 K>1 的逐步 draft-decode FULL |

Target 捕获时，`make_dummy` 在 `num_tokens // num_reqs > 1` 时把 dummy batch 改成 `SpecDecoding`。上游 dummy 默认是 `DecodeOnly`。抓错之后，replay 和运行时不是同一条 attention。

Runner 里的提升只覆盖均匀 verify：每个请求恰好 1 个有效 token，其余是 draft，且每个请求的 scheduled token 数等于 `decode_query_len`。不满足就保持 eager。`#17461` 之后共享 helper 会把 MTP verify 标成 `ChunkedPrefill`；310P 若不提升，FULL 每步都是 `CUDAGraphMode.NONE`。

Draft 捕获分两段：

- K=1 draft-prefill FULL 复用 target 的 batch 形状，`q_len = 2`。捕获时同样强制 `SpecDecoding`。若抓成 `DecodeOnly` / PA，replay 出的 draft 是坏的，接受率可以掉到大约 55%，rejection 之后最终文本仍可能看起来正确。
- K>1 的 draft-decode 不能把 CPU `slot_mapping` 的 D2H/H2D 录进 NPU GLOBAL 图。只捕获一个 `q_len = 1` 的 step；步与步之间在 host 上重算 slot，再逐 step `run_fullgraph`。

`resolved_tasks` 为空时，target 和 draft 都跳过 update stream 与 compute stream 的握手。不要为了保险加 `torch.npu.synchronize()`。小模型 MTP 的 TPOT 会先退。seq_lens、slot mapping 在 compute stream 上刷新，同流顺序足够。

当前校验放行后才能测的模型：Qwen3 dense、Qwen3 MoE（不开 EP）、Qwen3-VL / MRoPE、Qwen3.5 hybrid。Gemma 的 RMSNorm 公式不能外推成 Gemma4 MTP 已在 310P MRV2 放行。先过 `_validate_config` 和真实启动。

## 实操

一次只改一个变量。顺序是：eager 且 MTP 关 → eager 且 K=1 → eager 且 K>1 → 只开 `FULL_DECODE_ONLY` → 图和 MTP 同时开。前一层已经能分开两个候选就停。测量用 [aisbench-verification.md](aisbench-verification.md) 的 C、D、E。每个候选重启，不复用 KV 或 prefix cache。

启动前核对：

1. `speculative_config.method` 是 `"mtp"`，K 与 capture sizes 一致。K=1、batch N 的 sizes 要包含 `N * 2`；K>1 要包含 `N * (1+K)`。
2. 日志里出现 `Capturing 310P MTP draft ACLGraph`，以及 target 捕获没有退回父类的纯 DecodeOnly dummy。
3. 稳态 decode 出现一次 `ACL graph replay is active for the 310P draft model`。prefill 与 per-step decode 两条都要看到对应字样，不能只看到其中一条。
4. 稳态 trace 若仍是一长串 eager attention 小算子，先修 capture size 和 attention state，不换融合算子。
5. 性能数字把 TTFT、捕获阶段和稳态 TPOT 分开。开图后的第一次长 prompt 含捕获成本。
6. 报告平均接受长度。收益约等于少做的 target step，减去 draft 和 rejection 成本。接受率下降时先查 sampler 和图标签，不查 matmul。

温度实验单独做。`temperature >= 1e-5` 的请求会让整批走 CPU 概率实现。greedy 性能只用同质 `temperature=0` batch。

## 性能和精度的报错点

先对日志和接受率，再决定这是性能回退还是精度问题。最终文本正确不能证明 draft 图是对的。

| 现象 | 先看 | 含义 |
|---|---|---|
| 开了图，TPOT 仍等于 eager | capture sizes、`_force_eager_pc_batch`、`_force_eager_spec_batch`、`_promote_mtp_spec_decoding_attn_state` | 没 replay。非均匀 batch、prefix-cache hit，或 verify 没被提升回 `SpecDecoding` |
| 小模型 MTP 开图后更慢，时间线多出全流同步 | `ModelAclGraphManager310._updatable_graph_replay`、draft 的同名方法 | 空 `resolved_tasks` 仍做了 handshake，或有人加了 `synchronize()` |
| 捕获时报 `seq_lens must already be on-device` | `attention_v1.forward_paged_attention` | pageable H2D 发生在 `torch.npu.graph()` 内。把 `seq_lens` 移出 capture |
| 捕获时报 `draft slot_mapping cannot D2H while the NPU stream is capturing` | `AscendMTPSpeculator310._as_numpy_host` | K>1 的 host slot 计算进了图。应留在逐步 replay 之间 |
| 第一次 draft-prefill FULL replay `TypeError` | `AutoRegressiveAclGraphManager310._updatable_graph_replay` | `run_fullgraph` 丢了 `draft_attn_metadatas` |
| 平均接受长度约 0.55，最终答案仍大致正确 | draft `make_dummy` 的 `attn_state` | draft-prefill 抓成了 `DecodeOnly` / PA，replay 的 draft 是坏的，rejection 把错误吞掉了。这是精度问题，不要报成“性能差一点” |
| FULL 后首 token 乱码，eager 正常 | `Ascend310PModelState` 的 capture `seq_lens` 刷新 | replay 用了捕获时的 buffer 地址，运行时写入了另一块。要原位刷新，不能换 tensor |
| chunked prefill 把半截 prompt 写进已生成 token | rejection 的 `num_sampled` | chunked prefill 行的采样计数必须是 0 |
| 混合温度 batch 的 TPOT 突然变差 | `RejectionSampler310V2` 的 `any_random` | 只要一条请求非 greedy，整批走 CPU 概率路径 |
| TP>1 捕获中途失败或图张数很少 | event-id 与模型深度 | blocked。不要加 stream 或 event 去“多抓几张” |
| dense MTP1 每步都有大块 KV zero | `_needs_kv_cache_zeroing_310p` | KV zero 只在 Mamba、eagle block drop 且 K>1 时开启 |

`finish_requests` 里已有一次 `current_stream().synchronize()`，用来避免请求结束改写 CPU block table 时图还在读旧布局。不要在 replay 热路径再加一次。

## 模型

| 模型 | 性能上真正要看的路径 | 不要误用 |
|---|---|---|
| Qwen3 dense | attention NZ KV、`npu_rms_norm` / `npu_add_rms_norm`、target replay、draft replay、host 记账 | 不要开 EP |
| Qwen3 MoE | 上述项，加上 FP16 gating、routing v2、加载期 NZ、hidden 对齐 16 | 不要开 EP；group 门控不要塞进 softmax kernel |
| Qwen3-VL | dense 路径 + MRoPE 整块 position 拷贝 + 视觉 encoder | 视觉时间不要算进 decode TPOT；视觉 batch 通常进不了 decode 图 |
| Qwen3.5 / Qwen3.8 | hybrid：`l2norm_310p`、gated RMSNorm、conv/ssm 的 CPU 计划、MTP 的 target 与 draft 图 | 先按 checkpoint `architectures` 确认变体；chunk 与 recurrent 都要测；eps 必须除 dim |
| Gemma | `npu_rms_norm(x, 1+weight, eps)` | 不要把通用 V2 的 Gemma4 MTP 当成 310P 已放行 |

Qwen3.5 / 3.8 的线性注意力在 prefill 走 chunk、decode 走 recurrent。只压测其中一个会漏掉另一条 L2Norm 或 state 路径。

## 不支持就不要测性能

当前 `_validate_config` 会拒绝 MLA、sleep、PP、DP、prefill/decode context parallel、EP、非 MTP 投机和 KV transfer。它已放行 LoRA、prefix caching、Qwen3-VL / MRoPE 与 Qwen3.5 hybrid。这些边界会随代码演进，执行时以当前校验函数为准。真实失败记录异常原文，不要关掉检查来得到一个性能数。
