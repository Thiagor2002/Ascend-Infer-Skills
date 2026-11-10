# ais_bench 与分项性能验证

只在需要拉起服务、做简单 curl，或跑 ais_bench 性能对比时读取。本文的性能顺序是：拉起服务、简单 curl、随机合成性能测试、多模态图片压测。前一层已经能分开两个候选就停。

先用当前版本的 `vllm serve --help` 和 `ais_bench --help` / `--search` 核对参数。下面的命令和配置是诊断模板，字段名或默认值变了就以当前帮助和 `--search` 打印的文件为准，不要照抄过期 flag。

## 口径

对比两侧除目标变量外保持一致，并写进记录：

- 源码目录、commit、CANN、PTA、驱动；
- 权重、量化、TP、物理卡、`ASCEND_RT_VISIBLE_DEVICES`；
- `max-model-len`、`max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization`；
- `cudagraph_mode`、`cudagraph_capture_sizes`、`enable_npugraph_ex`、MTP 步数、prefix cache；
- chat template 与 `enable_thinking`；
- ais_bench 的 `batch_size`、`request_rate`、`max_out_len`、`RequestCount` 或 `--num-prompts`。

`enable_npugraph_ex=false` 是 310P 约束。`gpu-memory-utilization=0.6` 只是 RC 设备上的经验起点，改动必须记入对比。服务从确认过的 vLLM-Ascend 源码目录启动。每个候选重启，不复用 KV 或 prefix cache。不抢占其他任务占用的卡。

`request_rate` 是到达速率，不是实际并发。官方说明里小于 0.1 会一次把请求发出去。`batch_size` 是最大并发。二者分别记录，不要把其中一个写成另一个。

## 指标

实验开始前指定一个主指标。其余只做护栏，不临时更换主指标来宣布胜利。

| 指标 | 敏感项 |
|---|---|
| TTFT | prefill 长度、视觉 encoder、图捕获是否混进首包 |
| TPOT | host 气泡、图 replay、MTP 接受率、norm/attention |
| output throughput | 稳态 decode 与并发 |
| request throughput | 调度、并发、短请求 |

至少 3 次，报中位数和极差。warmup 不计入。单次数字不作为结论。OOM、图捕获失败、启动失败记 blocked。

MTP 必须同时报平均接受长度。接受率下降时先停性能结论，检查 rejection。文本随机合成的 TPOT 不要和图片压测的 TTFT 比。

## 层级

前一层已经能分开两个候选就停。不要用并发吞吐定位一个算子。

| 层级 | 配置 | 证明什么 |
|---|---|---|
| A | 单请求，temperature=0，短输出，eager，MTP 关 | 服务可跑 |
| B | 同 A，输出 256–1024 token | 稳态 TPOT，暴露 host 与 norm/attention |
| C | 只开 FULL_DECODE_ONLY，capture sizes 覆盖实际 batch | 图是否快于 eager decode |
| D | 只开 MTP，先 K=1 再 K>1，eager | draft + rejection 的收益 |
| E | 图与 MTP 同时开；token 数覆盖 `num_reqs * (1+K)` | 叠加是否掉回 eager |
| F | MoE 上重复 B | routing、dispatch、NZ 是否仍走已有 kernel |
| G | hybrid 上重复 B 与 D | L2、gated norm、Mamba align |
| H | 目标并发的 stream 性能 | 发布口径，不是定位口径 |

独立变量，不要合成一个开关：eager 与图、MTP 关与 K=1 与 K>1、prefix cache、量化、单个算子、capture sizes、`max_num_seqs`。

漏接 kernel 或元数据修复的最低服务证据是对应的 B/F/G 中位数。用户已经提供 trace 时，再核对目标小算子是否消失、已有 kernel 是否出现。H 只在用户要发布口径时跑。不要为了补这条证据去新采 profiler。

精度护栏至少是固定 prompt 的 token 对比，或用户指定数据集的小子集。L2Norm、gating、NZ、rejection 任一改动都要这条护栏。全量榜仅在用户要求时跑。精度用 `vllm_api_general_chat`：`temperature=0`，`ignore_eos=False`。性能用 `vllm_api_stream_chat`：`ignore_eos=True`。两套结果不能互相引用。

## 选择运行卡

若用户指定了物理卡号，按用户指定执行并记录，不要自行改选。若用户没有指定：

1. 先检查当前 shell 是否已有 `ASCEND_RT_VISIBLE_DEVICES`，不要静默覆盖继承的设备约束；
2. 查询所有可见 310P 的健康状态、占用进程和可用显存；
3. 按模型预估显存和 `-tp` 选择足够的空闲卡。空闲卡应无其他活动计算进程、状态正常，且可用显存满足启动；
4. TP 选择数量等于 TP degree 的卡，并考虑实际拓扑。不要把不同 rank 挤到同一张物理卡；
5. 启动前再确认一次占用。

```bash
echo "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
npu-smi info
```

确认物理卡号后，只在当前服务 shell 中设置：

```bash
# 单卡
export ASCEND_RT_VISIBLE_DEVICES=<physical-device-id>

# TP
export ASCEND_RT_VISIBLE_DEVICES=<physical-id-0>,<physical-id-1>
```

设置后，进程内逻辑编号可能重新映射。记录物理卡号、变量值和服务使用的逻辑 rank。不要把该变量写入用户 profile、镜像或全局启动脚本。没有满足条件的空闲卡时，报告各卡健康状态、占用和可用显存并等待用户处理。不得终止其他任务或抢占占用卡。

## 启动服务

服务拉起从用户确认的 vLLM-Ascend 源码目录执行。启动前记录 `pwd`、branch 和 `HEAD`。

```bash
cd <vllm-ascend-source-path>
export ASCEND_RT_VISIBLE_DEVICES=<selected-physical-device-ids>
vllm serve <weight-path> \
  --host 127.0.0.1 \
  --port 1025 \
  --served-model-name <model-name> \
  --trust-remote-code \
  --max-model-len <context-length> \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs <n> \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

- 310P 必须显式写 `--max-model-len`。省略它会让 attention mask 按上下文长度的平方占用显存。
- `mamba-ssm-cache-dtype`、`--dtype float16`、`-tp`、量化参数、MTP 只在该实验的目标变量或用户基线里出现。
- `cudagraph_capture_sizes` 的单位是 token，不是请求数。MTP 实验改成实际 token 数，例如 batch 4、K=1 时覆盖 8，而不是只写 4。
- 图模式本身是要隔离的变量。层级 A/B 先用 eager，不要把上面的 `FULL_DECODE_ONLY` 同时打开。
- 保存物理卡号、环境变量、完整命令，以及从进程启动到服务就绪的日志。

## 简单 curl

curl 的 host、port、model 必须与服务一致。这一层对应层级 A/B，不代替 ais_bench。重复至少 3 次，保存请求 JSON、原始流式响应、服务日志，以及首 token 时间和后续 token 间隔。warmup 不计入。

短输出只证明服务可跑。看稳态 TPOT 时把 `max_tokens` 放到 256–1024，并打开 `ignore_eos`，避免句子提前结束把输出长度切短。

```bash
curl -sS -N -X POST http://127.0.0.1:1025/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "从 1 连续数到 200，每个数字单独一行。"}],
    "max_tokens": 512,
    "temperature": 0,
    "stream": true,
    "ignore_eos": true
  }'
```

## Aisbench随机数性能测试

随机合成数据只用于性能，不能当精度结果。在 benchmark 仓库根目录执行。默认用 `synthetic_gen_string` 固定输入输出长度；`synthetic_gen_tokenid` 只有在要绕开字符串分词、直接喂 token id 时才用，并且模型配置里的 `path` 必须指向 tokenizer。

先定位当前版本的配置文件：

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_string --mode perf --search
```

性能模型配置写在 `--search` 给出的 `vllm_api_stream_chat.py`。需要改共享文件时先复制一份并记录 diff，跑完按用户要求恢复。与服务不一致的字段只改这一处：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat", 
        path="<weight-path>",
        model="<model-name>",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="127.0.0.1",
        host_port=2026,
        url="",
        max_out_len=1024,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.0,
            ignore_eos=True,
        ),
    )
]
```

`stream=True` 是服务化性能的前提。`max_out_len` 与下面 `Output` 的长度一致。层级 B 保持 `batch_size=1`、`request_rate=0`。层级 H 才提高 `batch_size`，并单独记录 `request_rate`。

固定长度写在 `--search` 给出的 `synthetic_gen_string.py`。把同一分布的最小值和最大值设成同一个数，避免长短请求混在一个 TPOT 里：

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 128,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 512, "MaxValue": 512},
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 512, "MaxValue": 512},
        },
    },
}
```

`RequestCount`、输入长度和输出长度按本层实验填写。数值应小于 2**20，且输入加输出不超过服务的 `--max-model-len`。分布扫描是另一个实验，不要和固定长度结论写在一起。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_string --mode perf --debug
```

性能数据为 TTFT、TPOT、OutputTokenThroughput，以及 `outputs/<timestamp>/performances/vllm-api-stream-chat/` 下的 csv、json。三次重复各保留一份目录。`ignore_eos=True` 时输出 token 数应等于 `max_out_len`；对不上就先停，不要把提前结束的样本算进 TPOT。

## 多模态图片压测

只在文本层级已经能跑，且问题涉及视觉 encoder、多模态 TTFT 或图片并发时做。图片压测不替代文本随机合成，也不把视觉时间算进 decode TPOT。

310P 的 Qwen3-VL 图片性能使用 `textvqa_gen_base64`：图片以 base64 进请求，数据集任务名是 `textvqa_gen_base64`。先搜索配置：

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen_base64 --mode perf --search
```

模型侧仍用上面的 `vllm_api_stream_chat` 性能配置。图片压测的 `max_out_len` 按样本答案长度设置，Qwen3-VL 8B 的 310P 性能用例是 512。`batch_size` 和 `request_rate` 继续单独记录；该用例是 `batch_size=1`、`request_rate=0`。

数据集配置写在 `--search` 给出的 `textvqa_gen_base64.py`。`path` 指向准备好的 jsonl，`image_type` 保持 `image_base64`。Qwen3-VL 8B 的 310P 性能用例使用 `vllm-ascend/textvqa-perf-1080p` 这一 1080p 子集，不使用配置文件里默认的 TextVQA 全量验证集。

```python
textvqa_datasets = [
    dict(
        abbr="textvqa",
        type=TEXTVQADataset,
        path="<textvqa-perf-jsonl>",
        image_type="image_base64",
        reader_cfg=textvqa_reader_cfg,
        infer_cfg=textvqa_infer_cfg,
        eval_cfg=textvqa_eval_cfg,
    )
]
```

默认 prompt 仍是数据集里的那条 TextVQA 模板。不要改问题文本，除非用户的基线本来就改过。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen_base64 \
  --mode perf --num-prompts <n> --debug
```

`--num-prompts` 按数据集顺序取前 n 条，不会打乱。Qwen3-VL 8B 的 310P 性能用例是 12 条；用户已有基线时沿用基线的条数。全量 TextVQA 只在用户要求发布口径时跑。

用户自备图片时改用 `mm_custom_gen`，不要把自备图片塞进 TextVQA 配置。jsonl 每行一张图：

```json
{"type": "image", "path": ["<media-root>/sample.jpg"], "question": "用一句话说明图片中的主要物体。", "answer": ""}
```

`mm_custom_gen.py` 里的 `path` 指向这份 jsonl，`mm_type` 用 `"path"`。服务必须带覆盖这些文件的 `--allowed-local-media-path`。多图样本才提高 `--limit-mm-per-prompt.image`。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets mm_custom_gen --mode perf --debug
```

图片压测的主指标默认是 TTFT 和端到端时延。输出 token 数不稳定时，不要把 TPOT 当成视觉优化的证据。OOM 记 blocked，先降 `max-model-len`、`max-num-seqs` 或图片张数，不把失败运行写成变慢。

## 记录

每次关键运行留下：

- 服务完整命令、日志、`pwd`、branch、HEAD、物理卡和逻辑 rank；
- benchmark commit、模型配置和数据集配置的 diff、实际命令；
- 三次原始打屏结果和 `performances/` 目录；
- 主指标的中位数和极差，以及被固定住的其他开关；
- 失败时的原始异常。OOM、capture 失败和启动失败不换算成 TPOT。
