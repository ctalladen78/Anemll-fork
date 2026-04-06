from huggingface_hub import snapshot_download
import os

os.makedirs("Anemll/reference_model", exist_ok=True)
snapshot_download(
    repo_id="anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4",
    local_dir="Anemll/reference_model",
    local_dir_use_symlinks=False
)
