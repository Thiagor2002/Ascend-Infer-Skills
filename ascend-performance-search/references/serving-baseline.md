# 服务化基线与自检

本文件是单实例（混部）服务的拉起协议。PD 分离的 P/D 实例在此基础上追加 `--kv-transfer-config` 与 proxy，见对应参考。

## 1. base 特性栈

"base" 指本轮比较的共同底座，叠加特性时它保持不变。本项目在 A5/950 + Qwen3.8-27B 上的 base 是：

| 项 | 取值 | 对应参数 |
|---|---|---|
| 权重量化 | mxfp8（W8A8） | `--quantization ascend`，权重包自带 |
| KV cache | C8 包才开 | `--kv-cache-dtype mxfp8` |
| 投机解码 | MTP3 | `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'` |
| 异步调度 | 开 | `--async-scheduling` |
| Decode 入图 | FULL_DECODE_ONLY | `--compilation-config` 中的 `cudagraph_mode` 与 `cudagraph_capture_sizes` |
| prefix cache | 开 | `--enable-prefix-caching --prefix-match-unit 16` |

换模型或换机型时 base 要重新定义并写进报告，不要沿用本表。

## 2. 可参数化的拉起脚本

把布局差异收进环境变量，脚本本体固定，这样每个测点可以保存一份快照并按哈希比对。本项目实际使用的形式：

```bash
#!/usr/bin/env bash
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES="${QWEN_DEVICES:-0}"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib64"
export VLLM_DISABLE_COMPILE_CACHE=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_PROC_BIND=false
export TOKENIZERS_PARALLELISM=false
export VLLM_SERVER_DEV_MODE=1

model="${QWEN_MODEL:-/home/weights/Qwen3.8-27B-w8a8c8-mxfp8}"
compilation_config='{"cudagraph_capture_sizes":[1,6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168,174,180,186,192],"cudagraph_mode":"FULL_DECODE_ONLY"}'
if [[ -n "${QWEN_COMPILATION_CONFIG:-}" ]]; then compilation_config="$QWEN_COMPILATION_CONFIG"; fi

args=(serve "$model" --served-model-name qwen3.8
  --host 127.0.0.1 --port "${QWEN_PORT:-18100}"
  --data-parallel-size "${QWEN_DP:-1}" --tensor-parallel-size "${QWEN_TP:-1}"
  --max-model-len 133120 --max-num-batched-tokens "${QWEN_BATCH_TOKENS:-16384}"
  --max-num-seqs "${QWEN_MAX_SEQS:-128}" --gpu-memory-utilization 0.95
  --enable-prefix-caching --prefix-match-unit "${QWEN_PREFIX_MATCH_UNIT:-16}"
  --compilation-config "$compilation_config"
  --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
  --trust-remote-code --quantization ascend --async-scheduling
  --allowed-local-media-path / --mm-processor-cache-gb 0
  --mm-encoder-tp-mode data --mm-processor-cache-type shm
  --additional-config '{"enable_cpu_binding":true}')

if [[ "$model" == *w8a8c8* ]]; then args+=(--kv-cache-dtype mxfp8); fi
if [[ "${QWEN_DCP:-1}" != 1 ]]; then args+=(--decode-context-parallel-size "$QWEN_DCP"); fi
if [[ "${QWEN_PCP:-1}" != 1 ]]; then args+=(--prefill-context-parallel-size "$QWEN_PCP"); fi
if [[ -n "${QWEN_KV_CONFIG:-}" ]]; then args+=(--kv-transfer-config "$QWEN_KV_CONFIG"); fi

printf 'REPRO_COMMAND: '; printf '%q ' vllm "${args[@]}"; printf '\n'
exec vllm "${args[@]}"
```

把展开后的 `REPRO_COMMAND` 落到该测点目录的 `serve-command.sh`，脚本本体落 `serve-script-snapshot.sh`。报告里不要只写"已开 Decode 入图"，必须给出完整的 `--compilation-config`，因为捕获尺寸列表直接决定拐点位置。

`--max-model-len` 要覆盖最长场景的输入 + 输出 + 余量。128k/1k 场景下 133120 = 131072 + 1024 + 1024。

## 3. 图捕获上限（最常见的假拐点）

投机解码开启后，decode 一步里每个请求提交 `num_speculative_tokens + 1` 个 token。vLLM 的 `adjust_cudagraph_sizes_for_spec_decode` 按这个倍数对齐捕获尺寸，未被覆盖的批量回退 eager。

```text
可入图的最大请求数 = max(cudagraph_capture_sizes) / (num_speculative_tokens + 1)
```

MTP3、上限 192 token 时是 48 个请求。本项目四卡 TP4、C8、4k/2k 的实测：并发 48 时 TPOT 12.7ms，并发 52 时 72.0ms，并发 64 时 72.3ms。这不是硬件饱和，是回退 eager。

拉起后在服务日志中核对 `max_cudagraph_capture_size`，把它换算成请求数记进 `job.json`。如果寻优的目标并发可能超过它，另起一个只改捕获范围的变体（如扩到 512 token）做对照，base 变体保持原参数不动。

## 4. 启动后自检

按顺序做完再压测，任一步失败就停下记 blocked：

```bash
# 1) 健康与模型名
curl -s http://127.0.0.1:${PORT}/v1/models

# 2) 纯文本一次性推理，确认能出 token
curl -s http://127.0.0.1:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8",
       "messages":[{"role":"user","content":"Who won the world series in 2020?"}],
       "max_tokens":32}'

# 3) 流式，确认 SSE 分片真的逐 token 返回（TPOT 依赖流式时间戳）
curl -sN http://127.0.0.1:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8","stream":true,
       "messages":[{"role":"user","content":"count to ten"}],
       "max_tokens":32}'

# 4) 计数器可读，后面算命中率要用
curl -s http://127.0.0.1:${PORT}/metrics | grep prefix
```

走代理环境时先 `unset http_proxy https_proxy`，否则 `127.0.0.1` 的请求会被代理吞掉，表现为 curl 不通但服务正常。

多 DP 混部时，`X-data-parallel-rank` 头可以把请求定向到指定 DP 域，自检和前缀预热都要逐域做一遍。这个头不能发给 PD 分离的 proxy。

## 5. 显存与块数

拉起日志里的 KV cache block 数 × block_size（128 token）是该实例能容纳的 token 总量。用它反推最大可并发请求数：

```text
可容纳请求数 ≈ block 数 × 128 / (输入长度 + 输出长度)
```

寻优的并发上界不要超过这个值，否则测到的是抢占和重算，不是稳态吞吐。显存不足导致的 OOM 把该候选判为无效，不记成一个很差的性能点。

## 6. 停服

一个候选测完就停进程再起下一个。只停本轮自己拉起的 PID，确认创建时间与端口归属后再动手；不要按进程名或端口模糊 kill，不要重置 NPU，不要动别人的容器。
