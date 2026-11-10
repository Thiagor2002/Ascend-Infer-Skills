# ais_bench 性能验证

设计或执行修复前后的服务性能对比时读取。本文只验证已经指出的性能回退或明显代码问题，不用来设计新融合，也不用来采集 profiler。

顺序固定为：选择空闲卡、拉起服务、简单 curl、随机合成性能测试。只有 Qwen3-VL 或其他视觉模型才继续做图片压测。前一层已经能分开两个候选就停。没有 310P 时不要把本文的命令当成已经跑过。

先用当前版本的 `vllm serve --help` 和 `ais_bench --help` / `--search` 核对参数。命令和配置是模板。flag 或字段名变了，以当前帮助和 `--search` 打印的文件为准。

硬件限制见 [hardware-constraints.md](hardware-constraints.md)。图模式、MTP 和模型路径见 [graph-mtp-models.md](graph-mtp-models.md)。精度数据集和 L0–L5 以精度技能的 `accuracy-testing.md` 为准。

## 口径

一次实验只有一个目标变量。其余开关在两侧保持一致，并写进记录：

- vLLM 与 vLLM-Ascend 的目录、branch、commit；运行时 import 路径；CANN、PTA、驱动；
- 权重、量化、TP、物理卡、`ASCEND_RT_VISIBLE_DEVICES` 和逻辑 rank；
- `max-model-len`、`max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization`；
- `cudagraph_mode`、`cudagraph_capture_sizes`、`enable_npugraph_ex`、MTP 步数、prefix cache；
- chat template 与 `enable_thinking`；
- ais_bench 的 `batch_size`、`request_rate`、`max_out_len`、`RequestCount` 或 `--num-prompts`。

`enable_npugraph_ex=false` 是 310P 约束，除非用户给出该版本已经验证过的其他设置。没有用户基线时，RC 设备从 `gpu-memory-utilization=0.6` 起；沿用已有服务时用该服务的值。改这个值就是一次新实验。

服务从确认过的 vLLM-Ascend 源码目录启动。每个候选重启进程，不复用 KV cache 或 prefix cache。不抢占其他任务占用的卡。输入长度、输出长度和并发是固定条件，不能为了让数字好看而临时缩短。

`request_rate` 是到达速率，不是实际并发。小于 0.1 会一次把请求发出去。`batch_size` 是最大并发。二者分别记录。

## 指标与硬门槛

实验开始前指定一个主指标。其余只做门槛，不临时更换主指标来宣布胜利。

| 指标 | 敏感项 |
|---|---|
| TTFT | prefill 长度、视觉 encoder、图捕获是否混进首包 |
| TPOT | host 气泡、图 replay、MTP 接受率、norm/attention |
| output throughput | 稳态 decode 与并发 |
| request throughput | 调度、并发、短请求 |

一次可引用的结果同时满足：

- 至少 3 次，报中位数和极差；warmup 不计入；单次数字不作为结论；
- 失败请求数为 0；
- `ignore_eos=True` 时，输出 token 数等于 `max_out_len`；对不上就停，不把提前结束的样本算进 TPOT；
- OOM、图捕获失败、启动失败记 blocked，不记成变慢；
- MTP 同时报平均接受长度。接受率下降时先停性能结论，检查 rejection。

文本随机合成的 TPOT 不要和图片压测的 TTFT 比。单请求 TPOT 不要写成并发吞吐。`ignore_eos=True` 的性能结果不能当作精度证据。

## 层级

| 层级 | 怎么跑 | 证明什么 |
|---|---|---|
| A | 下面的短 curl：temperature=0，eager，MTP 关 | 服务可跑 |
| B | 长 curl，或 `batch_size=1` 的固定长度随机合成；输出 256–1024 token | 稳态 TPOT |
| C | 只把编译配置改成 FULL_DECODE_ONLY，capture sizes 覆盖实际 batch | 图是否快于 eager decode |
| D | eager 上只开 MTP，先 K=1 再 K>1 | draft 与 rejection 是否拖慢 |
| E | 图与 MTP 同时开；capture 覆盖 `num_reqs * (1+K)` 个 token | 叠加后是否退回 eager |
| F | MoE 上重复 B | routing、dispatch、加载期 NZ 是否仍走已有 kernel |
| G | hybrid 上重复 B 与 D | L2、gated norm、Mamba 计划是否仍在 |
| H | 只提高 `batch_size`，`--num-prompts` 仍为 `4 × batch_size`，长度仍是 1024 | 发布口径，不是定位口径 |

独立变量不要合成一个开关：eager 与图、MTP 关与 K=1 与 K>1、prefix cache、量化、单个代码修复、capture sizes、`max_num_seqs`。

漏接 kernel 或元数据修复的最低服务证据是对应的 B、F 或 G 中位数，外加固定 prompt 的 token 对比。H 只在用户要发布口径时跑。不要为了补证据去新采 profiler。

按模型选择数据集，不要用一个模型的数字代表另一个：

| 模型 | 性能验证 | 不要做 |
|---|---|---|
| Qwen3 dense | A → B；需要看图时再做 C | 不要开 EP |
| Qwen3 MoE | 在 dense 的层级上重复 B，确认 gating 与 NZ | 不要开 EP，不要把 group 门控塞进 softmax kernel |
| Qwen3-VL | 文本 B 通过后，再做单图 curl 和图片压测 | 不要把视觉 TTFT 写成 decode TPOT |
| Qwen3.5 / Qwen3.8 | 先确认 checkpoint 变体，再做 B；开了 MTP 再做 D | 不要只压 chunk 或只压 recurrent |
| Gemma | 文本 A/B；先确认 310P 真能启动 | 不要把通用 V2 的 Gemma4 MTP 当成 310P 已放行 |

## 选择运行卡

用户指定了物理卡号就按指定执行并记录，不要自行改选。没有指定时：

1. 先看当前 shell 是否已有 `ASCEND_RT_VISIBLE_DEVICES`，不要静默覆盖；
2. 查询可见 310P 的健康状态、占用进程和可用显存；
3. 按模型显存和 TP 选择空闲卡。空闲卡没有其他计算进程，状态正常，显存够启动；
4. TP 使用数量等于 TP degree 的卡，不要把不同 rank 挤到同一张物理卡；
5. 启动前再确认一次占用。

```bash
echo "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
npu-smi info
```

只在当前服务 shell 中设置：

```bash
# 单卡
export ASCEND_RT_VISIBLE_DEVICES=<physical-device-id>

# TP
export ASCEND_RT_VISIBLE_DEVICES=<physical-id-0>,<physical-id-1>
```

设置后进程内逻辑编号可能重新映射。记录物理卡号、变量值和服务使用的逻辑 rank。不要把该变量写入用户 profile、镜像或全局启动脚本。没有空闲卡时报告各卡状态并等待用户处理。不得终止其他任务或抢占占用卡。

## 启动服务

在用户确认的 vLLM-Ascend 源码目录启动。启动前记录 `pwd`、branch、HEAD，以及 Python 实际 import 到的包路径。`--dtype`、`-tp`、量化、MTP 和 `mamba-ssm-cache-dtype` 只在用户基线或本次目标变量里出现。310P 必须显式写 `--max-model-len`。

层级 A/B/D/F/G 使用 eager。下面的命令不打开图：

```bash
cd <vllm-ascend-source-path>
export ASCEND_RT_VISIBLE_DEVICES=<selected-physical-device-ids>
vllm serve <weight-path> \
  --host 127.0.0.1 \
  --port 2026 \
  --served-model-name <model-name> \
  --trust-remote-code \
  --max-model-len <context-length> \
  --gpu-memory-utilization 0.6 \
  --max-num-seqs <n> \
  --enforce-eager \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

只在层级 C/E 把 `--enforce-eager` 换成图配置。capture size 的单位是 token，不是请求数。纯 decode 的 batch 4 写 4；MTP K=1 的 batch 4 写 8。尺寸对不上会整步退回 eager。

```bash
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
```

保存物理卡号、环境变量、完整命令，以及从进程启动到服务就绪的日志。启动失败记 blocked，不要关掉校验去换一个性能数。

图片压测在文本服务命令上增加图像限制。纯图片服务关掉视频。`file://` 图片必须落在 `--allowed-local-media-path` 内；base64 的 TextVQA 不依赖这个路径。

```bash
  --allowed-local-media-path <media-root> \
  --limit-mm-per-prompt.image <images-per-request> \
  --limit-mm-per-prompt.video 0 \
```

`<images-per-request>` 与样本一致。单图压测写 1。

## 简单 curl

curl 的 host、port、model 必须与本次服务一致。这一层对应 A/B，不代替 ais_bench。重复至少 3 次。保存请求 JSON、原始流、服务日志、首 token 时间和后续 token 间隔。

短输出只证明服务可跑：

```bash
curl -sS -N -X POST http://127.0.0.1:2026/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "用一句话回答：1+1 等于几？"}],
    "max_tokens": 32,
    "temperature": 0,
    "stream": true
  }'
```

稳态 TPOT 与后面的 ais_bench 一样把输出固定为 1024，并打开 `ignore_eos`，避免句子提前结束把长度切短：

```bash
curl -sS -N -X POST http://127.0.0.1:2026/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "从 1 连续数到 400，每个数字单独一行。"}],
    "max_tokens": 1024,
    "temperature": 0,
    "stream": true,
    "ignore_eos": true
  }'
```

视觉模型在图片压测前先发一条单图请求。这条 TTFT 含视觉 encoder，不能当成文本 decode 的 TPOT。

```bash
curl -sS -N -X POST http://127.0.0.1:2026/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "file://<media-root>/sample.jpg"}},
        {"type": "text", "text": "用一句话说明图片中的主要物体。"}
      ]
    }],
    "max_tokens": 128,
    "temperature": 0,
    "stream": true
  }'
```

当前版本若拒绝 `ignore_eos` 或 `file://`，记下原始报错，改用该版本 `--help` 里存在的字段。不要同时改服务参数继续测。

## 随机合成性能测试

随机合成只用于性能，不能当精度结果。在 benchmark 仓库根目录执行。定位用 `synthetic_gen_string` 固定输入输出长度。`synthetic_gen_tokenid` 只在需要绕开字符串分词、直接喂 token id 时使用，并且模型配置的 `path` 必须指向 tokenizer。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --num-prompts <4 x batch_size> -w <output_dir>
```

`--num-prompts` 一般设为 `4 × batch_size`。层级 B 的 `batch_size=1` 时就是 4。`-w` 指定本次输出目录。

性能模型配置写在 `--search` 给出的 `vllm_api_stream_chat.py`。需要改共享文件时先复制并记录 diff，跑完按用户要求恢复。`host_ip`、`host_port`、`model` 必须等于本次 `vllm serve`。下面这个模板使用 `host_port=2026`、`max_out_len=1024`、`batch_size=1`。层级 B 保持 `batch_size=1`、`request_rate=0`，因此 `--num-prompts` 为 4。层级 H 才提高 `batch_size`，`--num-prompts` 仍取 `4 × batch_size`，并单独记录 `request_rate`。

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

`stream=True` 是服务化性能的前提。`max_out_len=1024`，与下面 `Output` 的固定长度相同。

固定长度写在 `--search` 给出的 `synthetic_gen.py`。同一分布的最小值和最大值设成同一个数，避免长短请求混在一个 TPOT 里：

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 128,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 1024, "MaxValue": 1024},
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 1024, "MaxValue": 1024},
        },
    },
}
```

`RequestCount=128`，输出长度固定为 1024，输入长度固定为 1024。数值应小于 2**20，且 输入输出总和不超过 `--max-model-len`。`--num-prompts` 取 `4 × batch_size`，并且不能大于 `RequestCount`。层级 H 只提高 `batch_size`，不改这两段长度。分布扫描是另一个实验。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf --num-prompts <4 x batch_size> -w <output_dir>
```

读取打屏的 TTFT、TPOT、OutputTokenThroughput，以及 `outputs/<timestamp>/performances/vllm-api-stream-chat/` 下的 csv 和 json。三次重复各保留一份目录。失败请求数不为 0 时，这次结果作废。

精度护栏另用 `vllm_api_general_chat`：`temperature=0`，`ignore_eos=False`。它和上面的性能配置不能互相引用。

## 多模态图片压测

文本层级已经能跑，且问题涉及视觉 encoder、多模态 TTFT 或图片并发时才做。图片压测不替代文本随机合成。

Qwen3-VL 的 310P 图片性能使用 `textvqa_gen_base64`。图片以 base64 进入请求。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen_base64 --mode perf --search
```

模型侧仍用上一节的 `vllm_api_stream_chat`。Qwen3-VL 8B 的 310P 性能用例把 `max_out_len` 设为 512，`batch_size=1`，`request_rate=0`。数据集配置在 `--search` 给出的 `textvqa_gen_base64.py`。`image_type` 保持 `image_base64`。`path` 指向准备好的 jsonl。该用例使用 `vllm-ascend/textvqa-perf-1080p` 这一 1080p 子集，不使用配置里默认的 TextVQA 全量验证集。

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

不要改数据集里的问题文本，除非用户基线本来就改过。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen_base64 \
  --mode perf --num-prompts <n> --debug
```

`--num-prompts` 按数据集顺序取前 n 条，不会打乱。Qwen3-VL 8B 的 310P 性能用例是 12 条。用户已有基线时沿用基线的条数。全量 TextVQA 只在用户要求发布口径时跑。

用户自备图片时改用 `mm_custom_gen`，不要把自备图片塞进 TextVQA 配置。jsonl 每行一张图：

```json
{"type": "image", "path": ["<media-root>/sample.jpg"], "question": "用一句话说明图片中的主要物体。", "answer": ""}
```

`mm_custom_gen.py` 的 `path` 指向这份 jsonl，`mm_type` 用 `"path"`。服务必须带覆盖这些文件的 `--allowed-local-media-path`。多图样本才提高 `--limit-mm-per-prompt.image`。

```bash
cd <benchmark-root>
ais_bench --models vllm_api_stream_chat --datasets mm_custom_gen --mode perf --debug
```

图片压测的主指标默认是 TTFT 和端到端时延。输出 token 数不稳定时，不要把 TPOT 当成视觉路径的证据。OOM 记 blocked，先降 `max-model-len`、`max-num-seqs` 或图片张数。

## 记录

每次关键运行留下：

- 服务完整命令、日志、`pwd`、branch、HEAD、import 路径、物理卡和逻辑 rank；
- benchmark commit、模型配置和数据集配置的 diff、实际命令；
- 三次原始打屏结果和 `performances/` 目录；
- 主指标的中位数和极差，失败请求数，以及被固定住的其他开关；
- MTP 实验的平均接受长度；
- 失败时的原始异常。OOM、capture 失败和启动失败不换算成 TPOT。

结论写明对应哪一层。未跑的层写未测。不要把这一次性能实验写成新的优化方案。
