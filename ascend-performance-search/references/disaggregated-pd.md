# PD 分离摸测与配平

分离指 prefill 和 decode 跑在不同实例上，中间用 proxy 分发、用 KV connector 传 KV。寻优目标仍是满足门限的最大单卡输出，但**旋钮和混部完全不同**，不要把混部的并发阶梯直接搬过来。

## 1. 拓扑

```text
AISBench ──> proxy ──┬──> Prefill 实例（kv_producer）
                     └──> Decode  实例（kv_consumer）
                              ▲
                  P ──KV 传输──┘
```

单卡输出吞吐按 **P + D 的全部 NPU 数**归一，proxy 不计卡。命中率是 P 侧属性，`POD_INFO` 填各 Prefill 节点及其 DP 域端口，不填 proxy。

## 2. 拉起

在 [服务化基线](serving-baseline.md) 的脚本上追加 `--kv-transfer-config`，P 与 D 只差 `kv_role`、`kv_port`、`engine_id` 和卡号：

```bash
# Prefill
export QWEN_DEVICES=0 QWEN_PORT=19101 QWEN_DP=1 QWEN_TP=1
export QWEN_KV_CONFIG='{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":30000,
  "engine_id":"<run>-prefill",
  "kv_connector_extra_config":{"use_ascend_direct":true,
    "prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
export VLLM_HOST_IP=127.0.0.1
exec bash serve-script-snapshot.sh

# Decode：QWEN_DEVICES=1 QWEN_PORT=19102，kv_role=kv_consumer，kv_port=30100，engine_id=<run>-decode
```

`kv_connector_extra_config` 里的 `prefill`/`decode` 的 dp/tp 必须同时写在 P 和 D 两侧，且与实际拉起一致。写错不一定立刻报错，可能表现为 KV 传输阶段挂住。

proxy 用 vLLM-Ascend 自带示例，不要自己写：

```bash
python3 /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 127.0.0.1 --port 19100 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 19101 \
  --decoder-hosts 127.0.0.1 --decoder-ports 19102 --workers 1
```

AISBench 压 proxy 端口。`X-data-parallel-rank` 头不能发给 proxy。

**多 DP 用多个独立 DP1 引擎。** 本项目的原生多 DP 与 MTP 子配置冲突，改为每个逻辑 DP 一个独立 DP1 引擎（各自的 NPU、API 端口、KV 端口），由同一个官方 proxy 分发。`job.json` 里记 `dp_replica_mode`，`status.json` 里记每个副本的 PID 与创建时间，脚本按 `prefill-command.sh`、`prefill-dp1-command.sh`、`decode-command.sh` 分别保存。只启动首个 P/D 副本不算完整拓扑。

## 3. 闭环并发不是分离场景的旋钮

这是本项目最重要的一条现场结论。2 卡 P1D1、64k/1k，闭环并发压到 128（搜索上限）时：

```text
TPOT 平均 13.7ms（门限 20ms，远未触及）
TTFT 平均 112474ms
状态 search_cap_reached
```

128k/1k 更极端：TPOT 8.8ms，TTFT 334596ms。TPOT 门限根本约束不住系统，请求全堆在 prefill 队列里，TTFT 变成唯一的真约束。继续加并发只会让 TTFT 线性上涨，测不出有意义的最优点。

所以分离场景要么同时给 TTFT 门限并以它为主约束，要么按下一节做 RPS 配平。只报一个 TPOT 达标的闭环点，等于没有结论。

## 4. RPS 配平流程

来自 950DT 的 PD 调测方法，适用于任意 mPnD。核心是：prefill 是生产者，decode 是消费者，系统吞吐由 decode 决定，prefill 的实例数由它要喂饱的 decode 决定。

参数分工：

| 位置 | 参数 | 作用 |
|---|---|---|
| AISBench | `request_rate` | 发送速率，应与 prefill 能力持平，喂满但不积压 |
| AISBench | `batchsize` | 发送队列长度，设得比 decode 能力大即可 |
| Prefill | `max_num_seqs` / `max_num_batched_tokens` | 单 DP 能力 ≈ min(max_num_seqs × 输入长度, max_num_batched_tokens)；实践取 8 / 16384 |
| Decode | `max_num_seqs` | 系统求解的关键 |
| Decode | `max_num_batched_tokens` | =（投机 token 数 + 1）× `max_num_seqs` |

五步：

**1）确认系统最低 TTFT 能否达标。** 1P1D，prefill `max_num_seqs=1`、`max_num_batched_tokens=16384`、`max_model_len=输入+1+1K`，decode 用基线配置（只用来组系统，不做 decode）。发送端 `request_rate=0`、`max_out_len=1`、`batchsize = prefill 并发数 × prefill DP 数`。这时 prefill 只处理一轮、没有积压，测到的就是系统最低 TTFT。不达标就先优化 prefill，不要往下走。

**2）压 prefill 的 RPS。** 同样 1P1D、`max_out_len=1`，prefill 改 `max_num_seqs=8`。这一步 TTFT 必然超标，目的只是拿到 prefill 的 RPS，作为后续 `request_rate` 的上界。

**3）定 decode 的 `max_num_seqs`。** 给太大 OOM，给太小测不出能力。先按目标吞吐、序列长度、prefill RPS 和 TPOT 要求估一个初值，然后 mP1D 拉起（prefill 数量给满，避免 prefill 成为瓶颈），看 decode 日志的 block 数：`block 数 × 128 token` 是能放下的 token 总量，换算出请求数。估算值超过它就要增加 decode 实例。

**4）压 decode 的 RPS。** 仍是 mP1D，输出用真实长度，`request_rate = 单 prefill DP 的 RPS × prefill DP 数 × prefill 实例数`，`batchsize ≥ decode 的 max_num_batched_tokens`。由实测 TPOT 和输出长度算出 decode 侧 RPS。

**5）配平。** 取 `m × rps_P = n × rps_D` 的整数解，按真实输入输出长度跑 mPnD，主要调发送端的 `request_rate` 和 `batchsize`。
- prefill 能力不足且没有多余卡：用第 4 步的结果，不硬凑；
- decode 少量不足：下调 `request_rate`；
- decode 差距较大：prefill 实例数 +1。

体感校准：4k/1k 这类用例，1K 输出在 decode 侧要 35s，而 TTFT 目标是秒级，decode 是绝对的耗时重点。正常运行时 P 和 D 是流水的——同一时刻有请求在 prefill、有请求在 decode，TTFT 不随时间上涨就说明 prefill 没积压。

## 5. 分离 vs 混部：什么时候值得

同一台 8 卡 A5 上的实测对比（单卡输出 tps，TPOT < 20ms）：

| 序列 | 分离 2 卡 P1D1 | 混部 2 卡最优 | 混部 1 卡 |
|---|---|---|---|
| 4k/2k | 1147 | 1257 (DP2-TP1) | 1260 |
| 16k/1k | 273* | 318 (DP2-TP1) | 334 |
| 64k/1k | 461* | 275 (DP1-TP2) | 363 |
| 128k/1k | 169* | 145 (DP1-TP2) | 177 |

带 \* 的点是闭环并发打到搜索上限 128 的结果，TTFT 已达 100 秒量级，只能当作"TPOT 维度下的容量上界"，不是可交付的工作点。即便如此也能看出方向：**输入越长，分离越划算**；短输出均衡场景分离反而不如混部，因为 KV 传输的开销换不回收益。

4 卡 P(DP1-TP2) + D(DP1-TP2) 的 4k/2k 是 737 tps/card（并发 48、TPOT 13.3ms），低于同卡数混部 DP4-TP1 的 1273，进一步印证短序列不要上分离。

## 6. 已知的启动失败

`P(DP2-TP1)+D(DP2-TP1)` 与 `P(DP1-TP2)+D(DP2-TP1)` 在当前版本直接 `server exited with code 1`。混合 Mamba/GDN 模型在 P/D 的 TP 不对称传输上有版本相关限制，依赖装齐（Mooncake、zmq、FastAPI、Uvicorn、aiohttp）不等于该组合可用。保留真实启动错误记 blocked，不要写成吞吐 0。

PD 分离叠加 PCP/DCP 的组合同样受 [混部矩阵](colocated-pd-matrix.md#6-特性叠加的实测结论) 里那条 `platform.py` 限制约束，当前版本不可用。

## 7. 分离场景的记录

布局名编码 P 和 D 两侧：`pd-c8-n4-Pdp1tp2-Ddp1tp2`。目录里保存 `prefill-command.sh`、`decode-command.sh`、`proxy-command.sh` 及三份日志，多副本再加 `-dp1` 等后缀。`job.json` 记 `p_dp`、`p_tp`、`d_dp`、`d_tp`、`kv_port`、`metrics_ports`、`dp_replica_mode` 和 `existing_pd_services`（每个后端的 PID、创建时间、卡、端口）。

压测前后都要核对 P、D、proxy 三者健康和 run ID 一致；只看 AISBench 退出码不够，KV 传输失败时请求可能静默退化。
