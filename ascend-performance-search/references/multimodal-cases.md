# 多模态 curl 与图文性能用例

典型用例是图文 10k/300，门限 `TTFT 平均 < 1000ms`、`TPOT 平均 < 20ms`、prefix cache 命中率 0%。和纯文本场景的区别：主约束是 TTFT 而不是 TPOT，输入里有图片，命中率必须压到零。

本文件的流程在本项目尚未实测（图文用例当时被推迟），内容来自工具文档与服务端参数的核对。按它执行时，第一次要把每一步的真实输出留档，不要直接引用这里的描述当作已验证。

## 1. 先把 curl 打通

不同服务后端接受的图片输入格式不同，用错格式会报参数错误或直接把图片当字符串处理：

| 格式 | vLLM | MindIE | SGLang | AISBench |
|---|---|---|---|---|
| 1. `"image_url": "demo.jpg"` | ✗ | ✓ | ✗ | ✓ |
| 2. `"image_url": {"url": "demo.jpg"}` | ✗ | ✓ | ✓ | ✓ |
| 3. `"image_url": {"url": "file:///abs/demo.jpg"}` | ✓ | ✓ | ✗ | ✓ |
| 4. `"image_url": {"url": "data:image/jpeg;base64,<b64>"}` | ✓ | ✓ | ✓ | ✓ |
| 5. `"image_url": {"url": "https://host/demo.jpg"}` | ✓ | ✓ | ✓ | ✓ |

**vLLM 只接受 3、4、5。** 本地图片走 `file://` 绝对路径，并且服务端必须带 `--allowed-local-media-path <目录>`（本项目 base 用的是 `/`）。

先确认纯文本通，再确认图文通：

```bash
# 纯文本（所有后端都支持）
curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"qwen3.8",
  "messages":[{"role":"system","content":"You are a helpful assistant."},
              {"role":"user","content":"Who won the world series in 2020?"}]}'

# 方式 3：本地文件，需 --allowed-local-media-path
curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"qwen3.8",
  "messages":[{"role":"user","content":[
     {"type":"text","text":"What is the text in the illustrate?"},
     {"type":"image_url","image_url":{"url":"file:///data/textvqa/train_val_images/1080p.jpg"}}]}]}'

# 方式 5：在线图片
curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"qwen3.8",
  "messages":[{"role":"user","content":[
     {"type":"text","text":"What is the text in the illustrate?"},
     {"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"}}]}]}'
```

方式 4 用脚本构造：

```python
import base64, requests

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b64 = encode_image("/data/textvqa/train_val_images/1080p.jpg")
payload = {
    "model": "qwen3.8",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "please describe this picture."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
    "max_tokens": 50,
}
r = requests.post("http://127.0.0.1:8095/v1/chat/completions", json=payload)
print(r.json()["choices"][0]["message"]["content"])
```

走代理环境先 `unset http_proxy https_proxy`，否则本机请求被代理吞掉。方式 5 反过来需要外网可达。

curl 不通就不要开始压测。图片路径必须是**服务端**可访问的路径，不是压测端的。

## 2. 服务端参数

base 服务已经带了多模态相关参数，图文用例沿用即可：

| 参数 | 作用 |
|---|---|
| `--allowed-local-media-path /` | 放行 `file://` 本地图片；收紧到数据集目录更安全 |
| `--mm-processor-cache-gb 0` | 关掉多模态预处理缓存。做性能摸测必须关，否则重复图片会被缓存命中，TTFT 偏低 |
| `--mm-processor-cache-type shm` | 缓存类型，`-gb 0` 时不生效 |
| `--mm-encoder-tp-mode data` | 视觉编码器按 data 切分 |

命中率 0% 的要求同时约束两处：prefix cache 和 mm processor cache。两个都要在结果里核对，不能只看 `/metrics` 里的 prefix 计数。

## 3. AISBench 图文测评

AISBench 覆盖文本图片（TextVQA）、文本视频（VideoBench）、文本音频（VocalSound）等理解类任务，多模态生成任务不支持。

环境与纯文本一致，建议独立 conda 环境：

```bash
conda create --name ais_bench python=3.10 -y && conda activate ais_bench
git clone https://gitee.com/aisbench/benchmark.git && cd benchmark/ && pip3 install -e ./
```

**第一步：配 API。** 以 OpenAI `/v1/chat/completions` 流式接口为例，改 `ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py`，四处必改：

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        model="qwen3.8",                       # 1. 等于 --served-model-name
        path="/home/weights/<权重目录>",         # 2. 权重路径，读 tokenizer
        max_seq_len=4096,
        request_rate=0,
        rpm_verbose=False,
        retry=2,
        host_ip="127.0.0.1",                   # 3. 服务 IP
        host_port=8100,                        # 4. 服务端口
        enable_ssl=False,
        max_out_len=300,                       # 图文 10k/300 用例的输出长度
        batch_size=128,                        # 最大并发
        generation_kwargs=dict(
            temperature=0.5, top_k=10, top_p=0.95, seed=None,
            repetition_penalty=1.03, ignore_eos=True,
        ),
    )
]
```

`ignore_eos=True` 保证输出定长，便于比较；这类结果不能当精度证据。

**第二步：配数据集。** vLLM 服务化用 `textvqa_gen_base64.py`，MindIE 用 `textvqa_gen.py` —— 对应第 1 节的格式支持差异，别配反。只需改数据集路径：

```python
textvqa_datasets = [
    dict(
        abbr='textvqa',
        type=TEXTVQADataset,
        path='ais_bench/datasets/textvqa_json/textvqa_val.jsonl',  # 改这里
        reader_cfg=textvqa_reader_cfg,
        infer_cfg=textvqa_infer_cfg,
        eval_cfg=textvqa_eval_cfg,
    )
]
```

`textvqa_infer_cfg` 的 prompt 模板用 `{'type': "image_text", 'data': ['image_url', 'text'], ...}`。

**第三步：跑。**

```bash
ais_bench --models vllm_api_stream_chat --datasets textvqa_gen --debug -m perf
```

`-m perf` 是性能模式，默认是精度模式。

三个容易踩的点：

1. 数据集目录下必须同时有 annotations 文件。传 `textvqa_val.jsonl` 时同目录要有 `textvqa_val_annotations.json`。
2. `textvqa_val.jsonl` 里的图片路径要**服务端**能访问到。
3. vLLM 用 `textvqa_gen_base64`，MindIE 用 `textvqa_gen`。

## 4. 10k/300 用例的寻优

和纯文本的区别只在约束和判据，搜索方法照 [并发寻优协议](concurrency-search.md)：

- **主约束是 TTFT。** 图片编码摊在首 token 之前，TTFT 1 秒的门限通常比 TPOT 20ms 先被打破。粗扫时两个指标都记，按先被突破的那个定区间。
- **并发起点要低。** 视觉编码器占算力，同等卡数下能承受的并发远低于纯文本 4k/2k。从 2、4、8 起。
- **命中率核对两处。** prefix cache 实测 0，且 `--mm-processor-cache-gb 0` 确认生效。数据集里图片重复会让编码结果被复用，必要时按请求换图。
- **输入长度口径。** 10k 指文本 + 图片 token 的合计，图片 token 数由分辨率和 patch 大小决定，报告里用服务端上报的实际平均输入，并把图片分辨率一起记下来——换一批图，输入长度就变了，结果不可比。

有效性校验沿用 [AISBench 压测](aisbench-harness.md#7-请求级有效性校验)，另加一条：每个请求的图片都被真正解析（响应内容与图片相关，不是把 URL 当文本回了一遍）。抽查几条响应正文，全是通用描述或答非所问时先回到第 1 节查格式。

## 5. 视频与音频

`video_url`、`audio_url` 的输入结构与 `image_url` 基本一致，格式支持矩阵同理。对应数据集是 VideoBench、VocalSound 等。切换时重新走第 1 节的 curl 验证，不要假设图片通了视频就通。
