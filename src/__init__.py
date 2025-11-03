import os
from huggingface_hub import login

# Only login if HF_TOKEN is set and valid
# If HF_TOKEN env var is already set, huggingface_hub will use it automatically
if "HF_TOKEN" in os.environ:
    token = os.environ["HF_TOKEN"]
    # Only attempt login if token looks like an actual token (starts with hf_)
    # Otherwise, huggingface_hub will use the env var automatically
    if token.startswith("hf_"):
        try:
            login(token=token)
        except Exception as e:
            # If login fails but HF_TOKEN is set, continue (it will be used automatically)
            pass
