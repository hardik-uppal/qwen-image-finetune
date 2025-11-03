# Qwen Vision-Language Model Gradio App

This Gradio application provides a user-friendly interface for interacting with Qwen Vision-Language models served through vLLM.

## Features

- 🖼️ **Image Upload**: Upload any image for analysis
- 💬 **Interactive Chat**: Ask questions or give instructions about the image
- ⚙️ **Configurable**: Adjust model parameters like temperature, max tokens, etc.
- 🚀 **Fast**: Powered by vLLM for high-throughput inference
- 🌐 **Web Interface**: Easy-to-use Gradio interface accessible via browser

## Quick Start

### Prerequisites

1. **vLLM installed**: Install vLLM if not already installed
   ```bash
   pip install vllm
   ```

2. **Dependencies**: Ensure required packages are installed
   ```bash
   pip install gradio openai pillow
   ```
   Or install from the project requirements:
   ```bash
   pip install -r requirements.txt
   ```

### Step 1: Start vLLM Server

First, start the vLLM server with a Qwen vision model:

```bash
# For Qwen2.5-VL-7B-Instruct (requires ~14GB VRAM)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000

# For Qwen2.5-VL-72B-Instruct (requires multiple GPUs)
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4

# With specific GPU(s)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

**vLLM Server Options:**
- `--host 0.0.0.0`: Make server accessible from all network interfaces
- `--port 8000`: Port number (default: 8000)
- `--tensor-parallel-size N`: Number of GPUs for tensor parallelism
- `--max-model-len`: Maximum sequence length (adjust for longer contexts)
- `--gpu-memory-utilization 0.9`: GPU memory utilization (default: 0.9)

### Step 2: Launch Gradio App

In a separate terminal, launch the Gradio interface:

```bash
cd /workspace/hardik/test_repos/qwen-image-finetune
python script/gradio_qwen_image_app.py
```

**Environment Variables (Optional):**
```bash
# Set custom vLLM endpoint
export VLLM_BASE_URL="http://192.168.0.20:8000/v1"
export VLLM_API_KEY="your-api-key-if-needed"

python script/gradio_qwen_image_app.py
```

### Step 3: Access the Interface

Open your browser and navigate to:
```
http://localhost:7860
```

Or if running on a remote server, use the public URL displayed in the terminal (if `share=True` is enabled).

## Usage Guide

### Basic Usage

1. **Upload an Image**: Click on the image upload area and select an image
2. **Enter a Prompt**: Type your question or instruction in the text box
3. **Click Generate**: Press the "Generate Response" button
4. **View Response**: The model's response will appear on the right side

### Example Prompts

- **General Description**: "Describe this image in detail"
- **Object Detection**: "What objects can you see in this image?"
- **Specific Questions**: "What color is the car in the image?"
- **Analysis**: "Analyze the composition and lighting in this photo"
- **Creative Tasks**: "Write a short story inspired by this image"

### Configuration Options

#### Model Configuration
- **vLLM Base URL**: URL of your vLLM server (default: `http://localhost:8000/v1`)
- **API Key**: Authentication key (if your server requires it)
- **Model Name**: The exact model name served by vLLM

#### Advanced Settings
- **Max Tokens**: Maximum length of the response (128-4096)
- **Temperature**: Controls randomness (0.0 = deterministic, 2.0 = very creative)
- **Max Edge Size**: Maximum image dimension before resizing (512-2048px)
- **JPEG Quality**: Compression quality for uploaded images (50-100)

## Using with Custom Fine-tuned Models

If you've fine-tuned a Qwen model using this framework, you can serve it with vLLM:

```bash
# Serve base model with LoRA adapters
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --enable-lora \
    --lora-modules my-lora=/path/to/lora/weights \
    --host 0.0.0.0 \
    --port 8000

# Then specify the LoRA adapter in the Gradio interface
# Model Name: my-lora
```

## Troubleshooting

### vLLM Server Issues

**Problem**: `CUDA out of memory`
```bash
# Solution: Reduce GPU memory utilization or use quantization
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --gpu-memory-utilization 0.7 \
    --quantization awq  # or gptq, if model supports it
```

**Problem**: Server not accessible from remote machine
```bash
# Solution: Check firewall settings and ensure host is 0.0.0.0
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000

# On remote machine, use:
export VLLM_BASE_URL="http://<server-ip>:8000/v1"
```

### Gradio App Issues

**Problem**: Connection error to vLLM
- Ensure vLLM server is running and accessible
- Check the Base URL in the configuration panel
- Verify network connectivity: `curl http://localhost:8000/v1/models`

**Problem**: Image too large
- Adjust "Max Edge Size" in Advanced Settings to resize images
- Reduce JPEG Quality to decrease file size

**Problem**: Response is cut off
- Increase "Max Tokens" in Advanced Settings

## Performance Tips

1. **Image Preprocessing**: Resize large images to 1024px or smaller for faster processing
2. **Batch Processing**: For multiple images, consider using the vLLM API directly
3. **GPU Selection**: Use specific GPUs with `CUDA_VISIBLE_DEVICES`
4. **Model Selection**: Use 7B models for faster inference, 72B for better quality

## Advanced: Remote vLLM Server Setup

For detailed instructions on setting up a remote vLLM server, see:
- [VLLM_REMOTE_SETUP.md](VLLM_REMOTE_SETUP.md)

## API Documentation

The app uses the OpenAI-compatible API provided by vLLM. For direct API usage:

```python
from openai import OpenAI
import base64

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

## Related Files

- `gradio_prompt_generator.py`: App for generating edit prompts from image pairs
- `generate_qwen25vl_prompts.py`: Batch processing script for prompt generation

## Support

For issues or questions:
- Check the main project [README.md](../README.md)
- Review [VLLM documentation](https://docs.vllm.ai/)
- Open an issue on the project repository


