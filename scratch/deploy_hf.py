import os
import sys

# Ensure huggingface_hub is installed
try:
    from huggingface_hub import HfApi
except ImportError:
    print("[INFO] Installing huggingface_hub package...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 1. Get your token from https://huggingface.co/settings/tokens (must be a WRITE token)
HF_TOKEN = "PASTE_YOUR_HUGGING_FACE_WRITE_TOKEN_HERE"

# 2. Enter your Hugging Face username
HF_USERNAME = "22alr047"

# 3. Enter the name you want for your Space
SPACE_NAME = "driver-drowsiness-api"
# ==============================================================================

if HF_TOKEN == "PASTE_YOUR_HUGGING_FACE_WRITE_TOKEN_HERE":
    print("[ERROR] Please replace the HF_TOKEN placeholder with your actual Hugging Face Write Token.")
    sys.exit(1)

repo_id = f"{HF_USERNAME}/{SPACE_NAME}"
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"[1/3] Connecting to Hugging Face and creating Space '{repo_id}'...")
api = HfApi(token=HF_TOKEN)

try:
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True
    )
    print(f"[SUCCESS] Repository '{repo_id}' created or already exists.")
except Exception as e:
    print(f"[ERROR] Failed to create repository: {e}")
    sys.exit(1)

print(f"[2/3] Uploading project files from '{project_dir}' to Space...")
try:
    # Exclude local build nodes, caches, and database logs to speed up upload
    api.upload_folder(
        folder_path=project_dir,
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[
            "**/__pycache__",
            "**/.git",
            "**/.gitattributes",
            "**/node_modules",
            "backend/drowsishield.db",
            "frontend/dist",
            "scratch/*"
        ]
    )
    print(f"[SUCCESS] Files successfully uploaded to Hugging Face Spaces!")
except Exception as e:
    print(f"[ERROR] Upload failed: {e}")
    sys.exit(1)

print("\n==========================================================================")
print("[3/3] Deployment Initiated!")
print("==========================================================================")
print(f"Hugging Face is building your container. Monitor build status here:")
print(f"👉 https://huggingface.co/spaces/{repo_id}")
print("\nOnce the build is complete, copy the App URL (ends with .hf.space)")
print("and paste it in your Vercel settings under Calibration -> API Settings!")
print("==========================================================================")
