# 310P 硬件约束

看护性能，或判断一段代码会不会在 310P 上变慢、启动失败时读取。先读当前仓库的硬件画像，再查下面的格式、page 和 event-id 限制。画像和校验入口会变，执行时打开对应符号核对，不要沿用本文的旧表。

本文只用于发现明显违反硬件约束的代码。不用于设计新融合算子。没有 310P 时只做静态对照，并标明未实测。

修复后的性能对比用 [aisbench-verification.md](aisbench-verification.md)。图和 MTP 的捕获差异见 [graph-mtp-models.md](graph-mtp-models.md)。漏接的是仓库里已经存在的 310P kernel 时，才读 [operator-substitution.md](operator-substitution.md)。

## 先确认画像

画像在 `vllm_ascend/device/hardware_profile.py` 的 `AscendDeviceType._310P`。`ascend310p1`、`ascend310p3`、`ascend310p5`、`ascend310p7` 以及 `ascend310p3vir0*` 都映射到这一档。业务代码用 `get_current_hardware_profile()`，不要在热路径里再写一套 SoC 名字分支。

`profile.supports(X)` 只表示契约 `X` 存在。没有某项 capability，表示调用方必须走该类自己写明的 fallback，或直接拒绝。不能据此推断“所有相关算子都不存在”，也不能把 910 画像上的 capability 抄进来。

当前 310P 画像选择这些实现，而不是 A2/A3/A5 的标准实现：

| 字段 | 310P 取值 | 代码落到 |
|---|---|---|
| `default_worker_cls` | `vllm_ascend._310p.worker_310p.NPUWorker310` | 310P worker，不是 `NPUWorker` |
| `attention_backend_family` | `COMPATIBILITY` | `AscendAttentionBackend310`。MLA、SFA、DSA 没有 310P backend |
| `device_adaptor_family` | `COMPATIBILITY` | `Ascend310PDeviceAdaptor` |
| `weight_layout_policy` | `FORCE_NZ` | 支持的 matmul 权重在加载后转 `FRACTAL_NZ` |
| `moe_comm_policy` | `ALLGATHER` | MoE 通信走 AllGather，不走 MC2 / MegaMoe |
| `quantization_backend_family` | `COMPATIBILITY` | 310P 量化配置，不走标准量化族 |
| `cpu_binding_mode` | `TOPO_AFFINITY` | 按拓扑绑核 |
| `device_addressing_mode` | `DIRECT` | 直接设备寻址，不是双芯片卡 |
| `requires_2d_nz_linear_input` | `False` | 不要套用 A5 的二维 NZ linear 限制 |

`Ascend310PDeviceAdaptor` 的差异：

- KV 写入用 `torch_npu._npu_reshape_and_cache`。`use_bnsd=True` 抛 `BNSD KV cache is not supported on Ascend 310P`。
- `index_fill_` 不可用，用 boolean mask 模拟。
- `clipped_swiglu` 是 clamp 和 sigmoid 的普通算子，不是融合 kernel。

## 画像登记的能力

310P 的 `capabilities` 只有下面八项。看护时，热路径应该落到右列的 310P 实现。

| capability | 启用的功能 |
|---|---|
| `COMPATIBILITY_OP_IMPLEMENTATIONS` | 注册 310P custom op：`SiluAndMul`、`RotaryEmbedding`、`MRotaryEmbedding`、`RMSNorm`、`GemmaRMSNorm`、`RMSNormGated`、`ParallelLMHead`、`VocabParallelEmbedding`、`MMEncoderAttention`、`Conv3dLayer`、`GatedDeltaNetAttention`、`MoERunner`、`RoutedExperts` |
| `FUSED_MOE_COMPATIBILITY` | MoE router 用 `AscendGroupedTopKRouter310` |
| `GDN_COMPATIBILITY` | Qwen3.5 GDN 的 `_forward_core` 和 `get_state_dtype` 换成 `AscendGatedDeltaNetAttention310` |
| `DISTRIBUTED_COMMUNICATION_ADAPTATION` | 安装 310P 的 `all_reduce` 适配 |
| `FUSED_SWIGLU_TUNING_ARGS` | 融合 dequant-SwiGLU-quant 可以带 `glu_alpha`、`glu_bias` |
| `RUNTIME_CUSTOM_OPS` | 运行时注册 vLLM-Ascend 自定义算子库 |
| `IRQ_CPU_RESERVATION` | 绑核时为主机中断保留 CPU |
| `RC_DEVICE_DISCOVERY` | `is_rc_device()` 区分 Root Complex 与 PCIe endpoint。Atlas 200I Pro 一类是 RC，Atlas 300I DUO 一类在 `lspci` 里能看到 accelerators |

## 不要从 910 画像搬过来

标准画像有、310P 画像没有的契约，热路径上出现就是缺陷或启动失败，不是待移植优化。常见的是：

| 没有的契约 | 不要做 |
|---|---|
| `TRITON_BATCH_MEMCPY`，以及任何 Triton kernel | 不在 310P 执行路径调用 Triton。`HAS_TRITON` 只说明符号可能被 import |
| `NPUGRAPH_EX` | `enable_npugraph_ex=false`，除非该版本另有验证记录 |
| `PAGED_ATTENTION` | 不用 A2/A3 的共享 paged-attention 契约。310P 走自己的 `AscendAttentionBackend310` |
| `NPU_TOP_K_TOP_P` | 采样不要依赖 `torch_npu.npu_top_k_top_p` |
| `GRAPH_QKV_NORM_ROPE_FUSION` 和其他 `GRAPH_*` | 这些 FX 融合含 Triton 或不在 310P 登记，不要为了对齐 910 打开 |
| `LORA_CUSTOM_OPS`、`BGMV_SGMV_META_REGISTRATION` | 不要假设 310P 有 A2 的 BGMV/SGMV kernel。LoRA 是否放行以当前 `_validate_config` 为准 |
| `FLA_GDN_PREFILL`、`CANN_MEGAMOE`、`MC2_*`、`MLA_FLASH`、`FP8_ATTENTION` | GDN 用 compatibility 实现；MoE 用 AllGather；MLA、MegaMoe、MC2、FP8 attention 不在这张画像里 |
| `ATB_EXTENSIONS`、`ATB_WARMUP`、`AUTO_ENABLE_CUSTOM_OPS` | 不要按 A2 的 ATB 预热和默认 custom-op 列表启动 |

Atlas 300I DUO 不支持 `triton` 和 `triton-ascend`。源码安装可能把这两个依赖带进来，运行前应卸掉。910 的 Triton 实现原样拷进 `_310p`，或靠 `HAS_TRITON` 才走通主路径，都是缺陷。没有对应的 `torch_npu` 或已落地的 310P 实现时，保持原有非 Triton 路径，并写明缺的是哪一个符号。

## FRACTAL_NZ

`FORCE_NZ` 只覆盖支持的 matmul 权重，在 `process_weights_after_loading` 转一次。Attention 的 K/V 使用私有 ACL NZ buffer。Mamba 状态保持连续 ND。

不是所有 tensor 都转 NZ：

- FP32 和 meta tensor 不转；
- matmul 权重的 k=1 或 n=1 时，`aclnnMatmulWeightNZ` 会拒绝，必须留在 ND。Qwen MoE 的 `shared_expert_gate` 属于这一类；
- MoE gating 的 logits 仍是 FP16 ND；
- BNSD KV cache 不支持。

两类格式问题：

1. 该在加载时完成的 NZ，出现在每个 token 的前向里。逐步 `npu_format_cast` 是代码问题。
2. 不该转 NZ 的权重或激活被转了。这会启动失败或走错 kernel，不要把它记成还没优化完。

## Paged attention

310P 不消费上游标准化 shared KV backing。`NPUModelRunner310V2.supports_standardized_shared_kv_backing = False`。

当前 runner 里的硬限制：

- `head_size <= 256`；
- `block_size * head_size <= 128 * 128`，并且 kernel block size 能整除该组每个 attention cache block；
- 上游默认 block size 16 不可用。hybrid 路径把 attention 与 mamba 的 page 对齐到 128，且 attention page 不小于 mamba page；
- 不支持不对称的 K/V head size；
- `seq_lens` 在进入 ACLGraph capture 之前必须已经在 device 上。capture 内 pageable H2D 直接失败，不能靠 `non_blocking=True` 绕过。

Prefill 和 chunked prefill 会物化一块 `[max_model_len, max_model_len]` 的 float16 因果掩码，再转成 FRACTAL_NZ。没有压缩掩码路径。省略 `--max-model-len` 让自动探测取到很长上下文时，这块掩码按平方占用显存并 OOM。OOM 记成 blocked，不记成 TPOT 变慢。

配置踩中上述限制时，先报告启动或 capture 失败的原文。不要放宽校验去换一个性能数字。

## Event-id

310P 的图模式受硬件 event-id 数量限制。当前发布说明只保证 `FULL_DECODE_ONLY`：

- TP=1 没有这条捕获数量限制；
- TP>1 时能捕获并 replay 的图数量有限，且随模型深度变化。文档中的例子是 Qwen3-32B 可以捕获并 replay 2 张图；
- MTP 的 target 图和 draft 图共用这份预算。捕获失败记 blocked。

多出来的 stream、event 或保险用的全流 `synchronize()` 都在消耗同一份 event-id，也会在小模型 MTP 上直接拉高 TPOT。没有 FIA/PA graph task 时不要补 handshake。capture 对不上实际 token 数时，整步退回 eager。先修 capture size，不新增 event。

## 怎么用

先指出违反了画像里的哪一条，或下面哪条硬限制。一条代码问题对应一处修改。

| 看到 | 处理 |
|---|---|
| 热路径 import 或启动 Triton，或打开 `NPUGRAPH_EX` / `GRAPH_*` | 去掉 310P 执行路径上的调用。没有已落地的 `torch_npu` 或 `_310p` 实现就保持原路径 |
| 前向里反复 ND↔NZ | 把合法的权重转换挪回加载阶段。k=1、n=1、FP32、gating logits 保持原格式 |
| 使用标准 paged-attention、MLA、BNSD 或 block size 16 | 先落到 `AscendAttentionBackend310` 和合法 page。失败运行记 blocked |
| capture 内搬 `seq_lens`，或新增 event / 全流同步 | 把元数据移出 capture，删掉多余同步 |
| MoE 走了 MC2、MegaMoe 或 EP | 310P 画像是 AllGather，且 MRV2 拒绝 EP。不要为了“优化”打开 |
| 热循环里 `.item()`、`.tolist()`、pageable H2D、列切片拷贝 | 这是明显的 host 开销，按单变量修复 |
