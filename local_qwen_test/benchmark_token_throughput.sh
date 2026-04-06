#!/bin/bash

# Token Throughput Test for localhost:8080 (OpenAI Chat Template)
set -e

BASE_URL="http://localhost:8080/v1"
MAX_TOKENS=512
NUM_RUNS=3
echo "=========================================="
echo "Token Throughput Benchmark: localhost:8080"
echo "Model: qwen (OpenAI format)"
echo "Max tokens: $MAX_TOKENS | Runs: $NUM_RUNS"
echo "=========================================="

declare -a ELAPSED_TIMES
declare -a TOKEN_ESTIMATES

test_throughput() {
  local run_num=$1
  echo -e "\n[Run $run_num] Starting..."
  
  time_start=$(date +%s.%N)
  
  RESPONSE=$(curl -s -X POST "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Generate 512 tokens of random text."}
      ],
      "max_tokens": 512,
      "temperature": 0.7
    }')
  
  time_end=$(date +%s.%N)
  elapsed=$(echo "$time_end - $time_start" | bc)
  response_size=$(echo "$RESPONSE" | wc -c)
  tokens_est=$((response_size / 4))
  
  ELAPSED_TIMES[$run_num]=$elapsed
  TOKEN_ESTIMATES[$run_num]=$tokens_est
  
  echo "  Response size: ${response_size} bytes"
  echo "  Estimated tokens: $tokens_est"
  echo "  Latency: $(printf '%.3f' $elapsed)s"
  
  if [ $(echo "$elapsed > 0" | bc) -eq 1 ]; then
    throughput=$(echo "scale=2; $tokens_est / $elapsed" | bc | sed 's/^\./0./' | xargs)
    echo "  Throughput: ${throughput} tokens/sec"
  fi
}

echo -e "\n=== Warmup Run ==="
test_throughput 0
echo "Warmup complete, proceeding with benchmark runs..."

for i in $(seq 1 $NUM_RUNS); do
  test_throughput $i
done

# Calculate averages
total_elapsed=0
total_tokens=0
for i in $(seq 1 $NUM_RUNS); do
  total_elapsed=$(echo "$total_elapsed + ${ELAPSED_TIMES[$i]}" | bc)
  total_tokens=$((total_tokens + ${TOKEN_ESTIMATES[$i]}))
done

avg_elapsed=$(echo "scale=3; $total_elapsed / $NUM_RUNS" | bc | sed 's/^\./0./')
avg_tokens=$((total_tokens / NUM_RUNS))

if [ $(echo "$avg_elapsed > 0" | bc) -eq 1 ]; then
  avg_throughput=$(echo "scale=2; $avg_tokens / $avg_elapsed" | bc | sed 's/^\./0./' | xargs)
else
  avg_throughput="N/A"
fi

echo -e "\n=========================================="
echo "FINAL RESULTS (Average)"
echo "Average latency: ${avg_elapsed}s"
echo "Average tokens: $avg_tokens"
echo "Token throughput: ~${avg_throughput} tokens/sec"
echo "=========================================="