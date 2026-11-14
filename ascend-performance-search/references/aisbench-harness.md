# AISBench 压测与有效性校验

压测端用 `aisbench_auto_tools_prefix`（封装 AISBench 的数据集生成、发送与命中率统计）。本文件覆盖配置、命令、预热和验收口径。

## 1. 环境

AISBench 建议与服务化环境隔离，避免依赖冲突：

```bash
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/ && pip3 install -e ./
```

`pip show ais-bench-benchmark` 的 location 就是工具里要填的 `WORK_PATH`。记录 AISBench 版本号，它决定了 `--num-warmups` 之类参数是否存在。

## 2. config.py

```python
DATASET_PATH = "/home/d00976699/dataset"   # 需预先 mkdir，绝对路径
WORK_PATH = "/home/d00976699/benchmark"    # benchmark 仓库目录
MODEL_NAME = "qwen3.8"                     # 必须等于 --served-model-name
MODEL_PATH = "/home/weights/Qwen3.8-27B-w8a8c8-mxfp8"  # 读 tokenizer
HOST_IP = "127.0.0.1"
HOST_PORT = "18100"                        # 混部填服务端口；PD 分离填 proxy 端口
DEFAULT_PERFORMANCE_TEST = "default_perf"  # 稳态测试才改 "stable_stage"
OUTPUT_DIR = "./outputs/qwen38_perf"
POD_INFO = []                              # 见下
```

`POD_INFO` 只影响命中率统计，不影响发送：

- PD 混部：填主节点 `ip:port`；多 DP 时填各 DP 域的 metrics 端口。
- PD 分离：填各 **Prefill** 节点及其 DP 域端口，不填 proxy。命中率是 P 侧的属性。

并行跑多个测点时，每个测点用独立的 `WORK_PATH` 副本和独立 `OUTPUT_DIR`，否则数据集缓存、`picked_ids.txt` 和输出会互相污染。

## 3. 命令

统一用闭环发送，并发是唯一的负载旋钮，总请求数固定为并发 × 4：

```bash
# 普通定长（4k/2k、16k/1k）
python3 aisbench_test.py --input_len 4096 --output_len 2048 \
  --data_num $((C*4)) --concurrency $C --request_rate 0 \
  --dataset_type normal --npu_num <卡数> --dp <DP>

# 长序列 + 90% 前缀（64k/1k、128k/1k）
python3 aisbench_test.py --input_len 65536 --output_len 1024 \
  --data_num $((C*4)) --concurrency $C --request_rate 0 \
  --dataset_type prefix_cache --repeat_rate 0.9 --prefix_num 1 \
  --prefix_test --seed 1 --npu_num <卡数> --dp <DP>
```

参数要点：

- `--request_rate 0`：一次性放满并发，闭环补发。PD 分离做 RPS 配平时才改用非零值。
- `--npu_num`：用于算单卡吞吐。混部填该实例卡数，分离填 P+D 全部卡数。
- `--dp`：保证前缀预热覆盖每个 DP 域。
- `--prefix_test`：先单独跑一遍前缀再跑全量，打屏才会有命中率。验收只看 `output_len` 等于目标长度的那一行，预热行不算数。
- 128k 把 `--input_len` 换成 131072，其余不变。

数据集结构是 `前缀 + 3 个随机 token（--seed 控制）+ 后缀`，`--repeat_rate` 是每条请求可命中的比例。

## 4. 普通用例必须零命中

同一数据源反复发送会让第二轮及以后意外命中 prefix cache，把 4k/2k 的 TTFT 和吞吐抬高。用每请求独立的 `cache_salt` 隔离（本项目通过 `no_prefix_api.py` 适配器注入），不要改动输入文本长度。验收时核对实测命中率为 0。

## 5. 长序列预热的对齐

90% 命中率是验收条件，预热错位会直接让它掉到个位数。三个常见错位来源：

1. **chat template**：模板会在正文前后追加 token，真实 prompt 比 `--input_len` 长，公共前缀的起点被整体平移。
2. **哈希块边界**：`--prefix-match-unit 16` 下，缓存以 16 token 为粒度匹配，预热文本末尾不足一块的部分不会成为可命中前缀。
3. **预热端点越界**：预热文本比正式请求的公共前缀多出尾部 token 时，缓存端点落在公共前缀之外。本项目 128k 用例需要把预热文本末尾的 1 个正文 token 去掉，正式数据集、目标长度和 `repeat_rate=0.9` 都保持原值。

用 `warmup_prefix.py` 做这个检查，每个测点保存实际调整量与源文件哈希，不要复用上一个测点的预热文本。

多 DP 混部时，正式测量前用 `X-data-parallel-rank` 逐域预热并保存每域的返回与耗时，否则未预热的域会拉低整体命中率。

## 6. 命中率独立复核

不要只信打屏数字。在正式测量开始前后各取一次计数器，自己算一遍：

```bash
curl -s http://{ip}:{port}/metrics | grep prefix
# hit rate = Δvllm:prefix_cache_hits / Δvllm:prefix_cache_queries
```

关键是"正式测量开始前"的那次快照要在预热**之后**取，否则预热命中会混进正式窗口。多节点时各 P 侧分别取，再和打屏的 ALL_PODS 行对齐。

## 7. 请求级有效性校验

一个测点只有全部通过下面几条，才允许进入最优排名：

| 校验项 | 判据 |
|---|---|
| 请求数 | 实际完成数 == 并发 × 4 |
| 成功率 | 每个请求都成功，无超时、无截断、无 5xx |
| 输出长度 | 每个请求的输出 token 数精确等于目标长度 |
| 流式证据 | 逐 token 时间戳存在（AISBench 输出的 SQLite 中）；工具的旧别名告警不能当作"非流式"的判据 |
| 普通用例命中率 | 实测 0 |
| 长序列命中率 | 89.5% ≤ 实测 < 90.5% |
| 门限 | TPOT 平均严格小于上限；有 TTFT 要求时同样严格小于 |

超限点、无效点、启动失败点都要保留原始记录，只是不参与排名。失败后重试另建 `r1`/`r2` 目录，不覆盖旧数据。

## 8. 结果读取

- `aisbench_result.csv`：本次性能结果；
- `aisbench.log` / `aisbench_all.log` / `outputs/<时间戳>/aisbench.log`：过程输出；
- `outputs/` 下的逐请求详情与流式时间戳 SQLite：有效性校验的依据。

结果表中的"平均输入"用服务端上报的实际值，与目标 `--input_len` 不等是正常的（chat template），两者都保留。

## 9. 常见故障

| 现象 | 处理 |
|---|---|
| `生成数据集失败，请清空 picked ids` | 删除 `picked_ids.txt` |
| 加载 tokenizer 报错 | 核对镜像内 transformers 与模型是否匹配 |
| `unrecognized arguments: --num-warmups` | 当前 AISBench 版本不支持，从 `aisbench_test.py` 中删除该参数 |
| 打屏不显示命中率 | 未加 `--prefix_test`，或 `POD_INFO` 配错，或 http_proxy 拦了 metrics 请求 |
