# 310P 硬件约束

看护性能或判断“这段代码会不会在 310P 上变慢”时读取。限制以当前 `vllm_ascend` 代码和文档为准；数字和校验入口会变，执行时打开对应符号核对，不要沿用过期表。

本文只用于发现明显违反硬件约束的代码。不用于设计新融合算子。

## 不支持 Triton

Atlas 300I DUO（310P）不支持 `triton` 和 `triton-ascend`。源码安装可能把这两个依赖带进来，运行前应卸掉。310P 的 hardware profile 没有 `TRITON_BATCH_MEMCPY` 这类 Triton 能力。

热路径上出现以下情况就是性能或可运行性问题，不是待做的优化课题：

- 310P 分支直接调用 Triton kernel，或靠 `HAS_TRITON` 才走通的主路径；
- 把 910 上的 Triton 实现原样拷进 `_310p`；
- 为了“对齐上游”在 capture 或每个 decode step 里引入 Triton。

`HAS_TRITON` 的兼容垫片只说明上游符号可能被 import。它不表示 310P 可以执行 Triton。没有对应的 `torch_npu` 或已落地的 310P 实现时，保持原有非 Triton 路径，并写明缺的是哪一个符号。

## FRACTAL_NZ

310P 的权重布局策略是 `FORCE_NZ`：支持的 matmul 权重在加载后转为 `ACL_FORMAT_FRACTAL_NZ`。Attention 的 K/V 使用私有 ACL NZ buffer。Mamba 状态保持连续 ND。

NZ 不是“所有 tensor 都转成 NZ”：

- FP32 和 meta tensor 不转；
- matmul 权重的 k=1 或 n=1 时，`aclnnMatmulWeightNZ` 会拒绝，必须留在 ND。Qwen MoE 的 `shared_expert_gate` 属于这一类；
- MoE gating 的 logits 仍是 FP16 ND，不要为了统一格式改成 NZ；
- BNSD KV cache 在 310P 上不支持。

看护时把格式问题分成两类：

1. 该在 `process_weights_after_loading` 完成的 NZ，出现在每个 token 的前向里。逐步 `npu_format_cast` 是代码问题。
2. 不该转 NZ 的权重或激活被转了。这会启动失败或走错 kernel，不要把它记成“还没优化完”。

## Paged attention

310P 使用自己的 paged attention，不消费上游标准化 shared KV backing。`supports_standardized_shared_kv_backing = False`。

当前代码里的硬限制：

- `head_size <= 256`；
- `block_size * head_size <= 128 * 128`，并且 kernel block size 能整除该组每个 attention cache block；
- 上游默认 block size 16 在 310P 上不可用。hybrid 路径要把 attention 与 mamba 的 page 对齐到 128，且 attention page 不小于 mamba page；
- 不支持不对称的 K/V head size；
- `seq_lens` 在进入 ACLGraph capture 之前必须已经在 device 上。capture 内做 pageable H2D 会直接失败，不能靠 `non_blocking=True` 绕过。

Prefill 和 chunked prefill 会物化一块 `[max_model_len, max_model_len]` 的 float16 因果掩码，再转成 FRACTAL_NZ。没有压缩掩码路径。省略 `--max-model-len` 让自动探测取到很长上下文时，这块掩码会按平方占用显存并 OOM。OOM 记成 blocked，不记成 TPOT 变慢。

配置踩中上述限制时，先报告启动或 capture 失败的原文。不要放宽校验去换一个性能数字。

## Event-id

310P 的图模式受硬件 event-id 数量限制。当前发布说明只保证 `FULL_DECODE_ONLY`：

- TP=1 没有这条捕获数量限制；
- TP>1 时能捕获并 replay 的图数量有限，且随模型深度变化。文档中的例子是 Qwen3-32B 可以捕获并 replay 2 张图；
- 310P profile 没有 `NPUGRAPH_EX`。`enable_npugraph_ex=false` 保持为约束，除非该版本另有验证记录。

看护时，多出来的 stream、event 或“保险用”的全流 `synchronize()` 都在消耗同一份 event-id，也会在小模型 MTP 上直接拉高 TPOT。没有 FIA/PA graph task 时不要补 handshake。capture 对不上实际 token 数时，整步退回 eager；先修 capture size，不新增 event。

## 对照代码时怎么用

先指出违反了上面哪一条，再决定要不要改。一条代码问题对应一处修改：

| 看到 | 处理 |
|---|---|
| 热路径 import 或启动 Triton | 去掉 310P 执行路径上的 Triton，改接已有 `torch_npu` 实现；没有现成实现就保持非 Triton 路径 |
| 前向里反复 ND↔NZ | 把合法的权重转换挪回加载阶段；k=1、n=1、FP32、gating logits 保持原格式 |
| head、block 或 page 不满足 paged attention | 先让配置落到合法 page，不把失败运行记成变慢 |
| capture 内搬 `seq_lens` 或新增 event/sync | 把元数据移出 capture，删掉多余同步 |
| 热循环里 `.item()`、`.tolist()`、pageable H2D、列切片拷贝 | 这是明显的 host 开销，按单变量修复；细节不变量需要时再查附录 |

修复后的性能对比用 [aisbench-verification.md](aisbench-verification.md)。漏接的是仓库里已经存在的 310P kernel 时，才读 [operator-substitution.md](operator-substitution.md)。
