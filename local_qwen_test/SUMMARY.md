# Local Qwen Model Test Summary

## Accomplishments

- Successfully connected to local qwen model server at `localhost:8080/v1`
- Ran token throughput benchmark with 512-token requests
- Measured average throughput of ~25 tokens/sec
- Fixed endpoint URL from `/v1` to `/v1/chat/completions`
- Fixed bash script variable scoping issues

## Challenges

- Initial endpoint URL was incorrect (`/v1` instead of `/v1/chat/completions`)
- Variable scoping in bash functions required using arrays to store results
- Byte-based token estimation (bytes/4) less accurate than server's tokenizer

## Results

| Run | Latency | Est. Tokens | Throughput |
|-----|---------|-------------|------------|
| Warmup | 30.97s | 785 | 25.35 t/s |
| 1 | 31.39s | 760 | 24.21 t/s |
| 2 | 29.36s | 745 | 25.37 t/s |

**Average: ~25 tokens/sec** (Server reported ~33 tokens/sec)