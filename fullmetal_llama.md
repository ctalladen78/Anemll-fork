
### Full metal build for llama.cpp

this is used to build llama.cpp with metal and amx support
benchmarks can be run with ./bin/llama-bench -m <model> -p <context_length> -n <num_tokens> -ctk <quant_type> -ctv <quant_type>
see baseline benchmark results in benchmark.log

```bash
cmake -B build \
  -DGGML_METAL=ON \
  -DGGML_AMX=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DCMAKE_APPLE_SILICON_PROCESSOR="apple-m5" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j

```