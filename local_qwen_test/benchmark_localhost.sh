#!/bin/bash

# Speed Test Script for localhost:8080
# Tests various endpoints and measures throughput, latency, and response times

set -e

BASE_URL="http://localhost:8080"
NUM_REQUESTS=1000

echo "========================================="
echo "Speed Test: localhost:8080"
echo "Requests per test: $NUM_REQUESTS"
echo "========================================="

# Test 1: Health check endpoint (GET)
echo -e "\n[TEST 1] Health Check GET /health"
time_start=$(date +%s.%N)
curl -s -o /dev/null -w "%{http_code} %{time_total}s" "$BASE_URL/health"
time_end=$(date +%s.%N)
echo
elapsed=$(echo "$time_end - $time_start" | bc)
echo "  Latency: ${elapsed:.3f}s (1000 requests) = ${(l)elapsed/1000}:$(((${(l)elapsed*1000})%1000)) ms"

# Test 2: Root endpoint (GET)
echo -e "\n[TEST 2] Root GET /"
time_start=$(date +%s.%N)
response=$(curl -s "$BASE_URL/")
time_end=$(date +%s.%N)
echo "  Response size: $(echo $response | wc -c) bytes"
elapsed=$(echo "$time_end - $time_start" | bc)
echo "  Latency: ${elapsed:.3f}s (1000 requests) = ${(l)elapsed/1000}:$(((${(l)elapsed*1000})%1000)) ms"

# Test 3: POST endpoint with JSON payload
echo -e "\n[TEST 3] POST /api/chat/completions (JSON)"
time_start=$(date +%s.%N)
curl -s -o /dev/null \
  -X POST "$BASE_URL/api/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"test"}]}' \
  -w "%{http_code} %{time_total}s"
time_end=$(date +%s.%N)
echo
echo "  HTTP Status: $(curl -s -o /dev/null -w '%{http_code}' "$BASE_URL/api/chat/completions")"
elapsed=$(echo "$time_end - $time_start" | bc)
echo "  Latency: ${elapsed:.3f}s (1000 requests) = ${(l)elapsed/1000}:$(((${(l)elapsed*1000})%1000)) ms"

# Test 4: High throughput benchmark (10k requests)
echo -e "\n[TEST 4] Throughput Benchmark (10,000 GETs)"
time_start=$(date +%s.%N)
for i in $(seq 1 $NUM_REQUESTS); do
    curl -s "$BASE_URL/health" > /dev/null &
done
wait
time_end=$(date +%s.%N)
echo "  Total time: ${time_end - $time_start:.3f}s"
echo "  Throughput: $(echo "scale=2; $NUM_REQUESTS / ($time_end - $time_start)" | bc) req/s"

# Test 5: Concurrent requests (10 concurrent)
echo -e "\n[TEST 5] Concurrent Requests (10 parallel)"
time_start=$(date +%s.%N)
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
curl "$BASE_URL/health" &
wait
time_end=$(date +%s.%N)
echo "  Total time: ${time_end - $time_start:.3f}s"
echo "  Latency (avg): ${(l)(time_end - $time_start)/10}:$(((${(l)(time_end - $time_start)*10)}%10)) ms"

echo -e "\n========================================="
echo "Speed Test Complete"
echo "========================================="