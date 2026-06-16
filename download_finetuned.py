import subprocess
import os

env = os.environ.copy()
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

local_dir = "/infini-data/checkpoints/openvla-7b-finetuned-libero-spatial"

print("Downloading openvla/openvla-7b-finetuned-libero-spatial...")
print(f"Destination: {local_dir}")

result = subprocess.run(
    ["hf", "download", "openvla/openvla-7b-finetuned-libero-spatial",
     "--local-dir", local_dir],
    env=env,
)

print(f"\nReturn code: {result.returncode}")
