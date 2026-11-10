# 漏接的已有 310P kernel

只在 [hardware-constraints.md](hardware-constraints.md) 已经表明热路径漏接了画像或 `_310p` 里落地的调用时读取。例如本应走 `npu_rms_norm` 却拆成 Pow、Mean、Rsqrt，或权重本应在加载时转 NZ 却在前向里 format cast。

本文把那一次调用接回去。不设计新融合，不从 profiler 里找下一个可替换算子，不替换已经接上且没有明显错误的 op。找不到受支持的现成调用就保持原实现。用户只要求解释时，给出漏接点和验证计划，不改代码。

## 什么时候核对

核对的前提是两件同时成立：

1. 当前 step 里的数学是固定归约或固定重排；
2. 仓库里已经有对应的 310P 或 `torch_npu` 调用。已有样例是 `npu_rms_norm`、`npu_add_rms_norm`、`npu_moe_gating_top_k_softmax`、`npu_moe_init_routing_v2`、`maybe_trans_nz`，以及画像登记的 `AscendRMSNorm310`、`AscendGroupedTopKRouter310`。

不在这里处理：attention 主 kernel 的内部实现、通信算法、调度策略、图捕获流程。那些走 [graph-mtp-models.md](graph-mtp-models.md) 或 [hardware-constraints.md](hardware-constraints.md)。也不要为了“还有更快算子”替换已经接上的 op。

## 怎么接回

1. 在本仓库搜同名数学的现有 310P 封装，只复用已落地的类或函数。
2. 在当前环境的 `torch_npu` 里确认符号存在，并记录 torch_npu 和 CANN 版本。不存在就淘汰该候选、保持原实现。不要编造函数名或参数。
3. 读仓库里的调用样例和测试，核对 dtype、连续性、ACL format、维度对齐、输出语义，以及能不能放进 ACLGraph。
4. 对照原式，写出必须改写的参数。已知约束：
   - L2 经由 RMSNorm：`weight = 1/sqrt(dim)`，`eps_rms = eps / dim`。
   - Gemma RMSNorm：缩放是 `1 + weight`，用 `AscendGemmaRMSNorm310`。
   - Qwen 有 residual 才能用 `npu_add_rms_norm`。
   - MoE gating：FP16 ND，单次调用 token 上限 1024，非 group 且无 correction bias。
   - MoE routing v2：hidden `% 16 == 0`。当前代码把 routing-v2 的 `expert_tokens` 直接交给 `group_list_type=0` 的消费者。用已知多 expert assignment 验证成对语义，不凭名字自行 `cumsum`。
   - NZ 放在 `process_weights_after_loading`，不放进前向。
5. 一个补丁只接回这一处。图捕获区域内不能新增数据相关的 Python 分支或 host 同步。捕获期形状不合法时，把该算子留在 eager 路径。

## 必须留下的回退

- grouped RMSNorm；
- grouped topk、custom routing、带 score correction 的门控；
- 不支持的 activation 按当前接口契约显式失败。只有上游要求支持且数值验证通过时才增加 native fallback；
- `temperature > 0` 的 rejection；
- `use_qk_l2norm_in_kernel=False`；
- 输入不满足对齐或 dtype 时的明确失败，或已有的 native 路径。不要静默 pad 出每步都在的临时 buffer，除非证明它比拒绝更便宜且数值一致。

## 验证

按顺序做，前一步失败就停：

1. 数值：与接回前的实现或 CPU 参考比 max abs 和余弦。覆盖 rank、dtype、group 开闭、`norm_before_gate` 两种顺序、chunk 与 recurrent。CPU UT 若 mock 了 `npu_rms_norm`，mock 必须实现同一公式，不能用空 MagicMock。
2. 单算子：目标 shape 至少 3 轮，报中位数。同步边界一致，不能让旧实现多一次 synchronize。
3. 服务： [aisbench-verification.md](aisbench-verification.md) 的 B、F 或 G。主指标用中位数。
4. 精度护栏：temperature=0 的固定 prompt token 对比。用户要求数据集时再跑数据集。
5. 并发吞吐只在用户要发布口径时跑。

用户已经提供 trace 时，再核对原来的小算子是否消失、已有 kernel 是否出现，以及有没有把同步或 format cast 挪到别的 step。没有 trace 时不要为了这一步去采集。

报告里分开写数值误差、TPOT 或吞吐。缺数值或服务证据时，不说“修复已验证”。
