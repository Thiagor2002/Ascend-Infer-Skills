# 用户手动性能寻优测试

这里是用户可以直接执行的最小验证序列，用来确认服务和压测链路可用，再交给 agent 做完整寻优。不要把这些命令当作性能结论。

## 1. 服务可用性

```bash
unset http_proxy https_proxy
PORT=18100

curl -s http://127.0.0.1:$PORT/v1/models
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
curl -sN http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8","stream":true,"messages":[{"role":"user","content":"count to ten"}],"max_tokens":32}'
curl -s http://127.0.0.1:$PORT/metrics | grep prefix
```

第三条必须看到逐段 SSE 返回。非流式下 TPOT 没有意义。

## 2. 图捕获上限核算

```bash
grep -i "cudagraph" <布局目录>/server.log | head
```

把 `max_cudagraph_capture_size` 除以（投机 token 数 + 1），得到能入图的最大请求数。并发越过它时 TPOT 会出现数倍突变。

## 3. AISBench 单点冒烟

```bash
cd <aisbench_auto_tools_prefix>
python3 aisbench_test.py --input_len 4096 --output_len 2048 \
  --data_num 16 --concurrency 4 --request_rate 0 \
  --dataset_type normal --npu_num 1 --dp 1
```

看 `aisbench_result.csv` 是否有 TPOT、TTFT、输出吞吐，`outputs/` 下是否有逐请求详情。

## 4. 前缀命中率冒烟

```bash
python3 aisbench_test.py --input_len 65536 --output_len 1024 \
  --data_num 8 --concurrency 2 --request_rate 0 \
  --dataset_type prefix_cache --repeat_rate 0.9 --prefix_num 1 \
  --prefix_test --seed 1 --npu_num 1 --dp 1
```

验收看 `output_len=1024` 的那一行，预热行不算。打屏命中率应在 90% 附近；明显偏低先查预热对齐，不要直接加大并发。

## 5. 固定并发扫描脚本

```bash
for c in 2 4 8 12 16 24 32; do
  python3 aisbench_test.py --input_len 4096 --output_len 2048 \
    --data_num $((c*4)) --concurrency $c --request_rate 0 \
    --dataset_type normal --npu_num 1 --dp 1
done
```

这只是粗扫。门限跨越区间内的细扫、请求级有效性校验和建档由 agent 按技能流程完成。
