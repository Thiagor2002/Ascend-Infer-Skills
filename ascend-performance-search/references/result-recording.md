# 建档与结果记录规范

寻优会产生几百个测点。没有固定目录结构和固定表头，结论无法复现，也无法判断某一行是不是已经过期。

## 1. 两级目录

```text
<ops-root>/<campaign>/              # 一轮摸测，如 test-qwen38-1113-stage1
├── matrix.json                     # 任务矩阵：布局 × 序列 × 特性
├── vllm-commit.txt / ascend-commit.txt / environment-packages.txt / lscpu.txt
├── weight-quantization-coverage.json
├── all-results.csv                 # 全量结果
└── <布局>/                          # 一个服务配置
    ├── job.json                    # 布局声明：模型、卡、DP/TP、端口、用例、变体
    ├── status.json                 # 运行状态、PID、当前用例
    ├── serve-command.sh            # 展开后的完整命令
    ├── serve-script-snapshot.sh    # 脚本本体快照
    ├── server.log
    └── in<输入>-out<输出>/
        ├── search.json             # 该序列的全部搜索点
        └── c<并发,3位补零>-r<重试号>/   # 一个测点
```

布局名要自解释，直接编码形态、模型、卡数与并行：`c8-n4-dp1-tp4-base`、`w8-n2-dp1-tp2-pcp1-dcp2`、`pd-c8-n4-Pdp1tp2-Ddp1tp2`。`c8`/`w8` 区分 KV cache 是否 C8，`n<N>` 是卡数，尾部是变体名。

PD 分离布局把 `serve-command.sh` 换成 `prefill-command.sh`、`decode-command.sh`、`proxy-command.sh` 及各自日志；多副本再加 `prefill-dp1-command.sh` 这类后缀，不能只保存首个副本就当作完整拓扑。

## 2. 测点目录

| 文件 | 用途 |
|---|---|
| `command.sh` | 该次 AISBench 的完整命令及工作路径 |
| `config.py` | 本测点使用的目的 IP/端口、模型路径、隔离的 WORK_PATH、输出位置 |
| `workspace/ais_bench/benchmark/configs/` | 本测点独立的 AISBench 模型与数据集配置 |
| `terminal.log`、`aisbench.log` | 完整过程输出 |
| `metrics-before-*.txt`、`metrics-after-*.txt` | 服务端计数器快照 |
| `metrics-measurement-before-*.txt` | 预热结束、正式测量开始前的计数器，用于独立复核命中率 |
| `warmup-prefix-adjustment.json`、`warmup_prefix.py` | 实际预热边界、调整量与源文件哈希 |
| `dp-prefix-priming.json` | 多 DP 逐域预热的真实返回与耗时（适用时） |
| `result.json` | 测量值、验收状态、请求级证据 |
| `outputs/` | AISBench 原始统计、逐请求详情、流式时间戳 SQLite |

重试用 `r1`、`r2` 新目录，保留失败目录原样。

## 3. 结果表

最终文档按形态和卡数分表：PD 混部 1/2/4/8 卡各一张，PD 分离 2/4/8 卡各一张。表头固定：

| 布局 | 序列长度 | Prefix Cache | 卡TPOT | 复现指导 | 上下文长度 | 平均输入 | 平均输出 | 总请求数 | 最大并发数 | 系统并发数 | 请求频率(req/s) | TTFT平均(ms) | TPOT平均(ms) | E2E时间(s) | 输出吞吐(tps) | 单卡输出吞吐(tps) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DP1-TP1 | 4k/2k | 0 | 20 | | 133120 | 4148 | 2048 | 64 | 16 | | 0 | | | | | |

格式约定：

- 序列长度写 `m k/n k` 形式，如 `4k/2k`、`128k/1k`；
- 上下文长度、平均输入、平均输出、总请求数、系统并发数是整数；
- `Prefix Cache` 填目标值（0 或 90），实测命中率另列或写进 `复现指导`；
- 平均输入用服务端上报的实际值，不是 `--input_len`；
- `复现指导` 指向模型包、卡配置和该行对应的脚本路径。

## 4. 并发曲线

每个（布局 × 序列）的完整并发扫描单独成表，不要塞进主结果表。用纯文本 Markdown 表分组存放并建索引文件，单文件控制在 240KB 以内。不生成图片。

## 5. 对照文档

特性对照单独成文，按布局、模型、卡数、序列对齐，比较各自达标的最高单卡输出。每一行注明两侧的最优并发不同，差值不是固定负载下的单算子收益。

## 6. 进度与恢复

长时摸测要有独立于对话的状态文件：

- 检查点：当前正在运行的任务、NPU 占用、端口、PID、当前用例、队列状态；
- 恢复说明：续跑前如何核对已有进程，哪些 PID 正在运行就不许重复启动或 kill；
- 复现文档：环境指纹、base 参数、AISBench 协议、有效性判据。

检查点由后台观察器定时刷新，和测试驱动分开。对话中断、额度不足时保留这些进程与文件，不因为对话结束就停掉正在跑的测点。

## 7. 边界

- 原始日志留在本机，不整包上传到对话，也不推送进 Git；日志可能含私网拓扑。
- 超限、无效、blocked 的记录全部保留，但不参与排名，也不混进结果表。
- 结果文件是自动生成的，手工修订要留痕，不要直接改汇总表的数字。
