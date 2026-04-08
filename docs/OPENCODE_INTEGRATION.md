# 🔗 Integrating Allama with OpenCode

Use OpenCode as a web UI to chat with your Allama models. OpenCode connects to Allama's OpenAI-compatible API.

---

## Quick Start

### Step 1: Start Allama Server
```bash
allama serve
```

Allama will start on `http://localhost:9000` with an OpenAI-compatible API at `http://localhost:9000/v1`.

### Step 2: Open OpenCode

**Option A: Use the Allama helper command**
```bash
allama ui opencode
```

This will:
- Verify Allama is running ✅
- Show you the configuration steps
- Try to open OpenCode in your browser (if running)

**Option B: Manual setup**
1. Open OpenCode in your browser
2. Go to **Settings** → **API Configuration**
3. Set the following:
   - **API Base URL:** `http://localhost:9000/v1`
   - **API Key:** `allama` (or any key, it's not validated)
   - **Model:** Select from the dropdown (will show all your Allama models)

### Step 3: Start Chatting

Once configured, OpenCode will show all your available Allama models:
- `Qwen3.5:27b`
- `Qwen3.5:35b`
- `Qwen3.5:35b-Code`
- `Qwen3vl:8b` (Vision)
- etc.

Select a model and start chatting! 💬

---

## Full Integration Guide

### Prerequisites

1. **Allama running:** `allama serve` on port 9000
2. **OpenCode installed/running:** Usually at `http://localhost:3000`

### Architecture

```
┌──────────────────┐
│   OpenCode UI    │  (Browser)
│  (localhost:3000)│
└────────┬─────────┘
         │ HTTP Request
         │ POST /v1/chat/completions
         │
┌────────▼─────────┐
│  Allama Server   │  (localhost:9000)
│  OpenAI-compat   │
└────────┬─────────┘
         │
    ┌────┴────┬────────┬────────┐
    │          │        │        │
    ▼          ▼        ▼        ▼
  vLLM      llama.cpp Remote  (More)
              (GGUF)  Provider
```

### Configuration Details

#### API Base URL

**What it is:** The root URL where Allama's API server is listening.

**Format:** `http://localhost:9000/v1`

**Components:**
- `http://` - Protocol
- `localhost` - Your machine (or IP address)
- `9000` - Default Allama port
- `/v1` - OpenAI-compatible API version

**To use on different machines:**
- Same machine: `http://localhost:9000/v1`
- Different machine: `http://192.168.1.100:9000/v1` (replace IP)

#### API Key

**What it is:** Authentication token for the API.

**For Allama:** Any value works (not validated in basic setup)
- `allama`
- `sk-test`
- `anything`

**Note:** If you add authentication later, configure it in `~/.allama/.env`

#### Model Selection

**How it works:**
1. OpenCode calls `GET /v1/models` on Allama
2. Allama returns all available models (local + dynamic/remote)
3. OpenCode displays them in the dropdown

**Available models depend on:**
- Logical models configured in `configs/logical/`
- Physical models configured in `configs/physical/`
- Dynamic remote models (created via `allama run ... on opencode`)

---

## Troubleshooting

### Problem: "Connection refused" when configuring OpenCode

**Solution:**
```bash
# Check if Allama is running
allama status

# If not running:
allama serve
```

### Problem: Models don't appear in OpenCode dropdown

**Solution:**
1. Verify Allama is serving models:
   ```bash
   allama list
   ```

2. Check if OpenCode can reach Allama:
   ```bash
   curl http://localhost:9000/v1/models
   ```

3. Try refreshing OpenCode in browser (F5)

### Problem: OpenCode can't send messages (timeout or error)

**Possible causes:**
- Model is loading (vLLM takes time to start)
- Wrong port configuration
- API Base URL is incorrect

**Solution:**
1. Check Allama logs:
   ```bash
   allama logs -f
   ```

2. Verify the model is loaded:
   ```bash
   allama ps
   ```

3. Test API directly:
   ```bash
   curl -X POST http://localhost:9000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen3.5:27b",
       "messages": [{"role": "user", "content": "hello"}]
     }'
   ```

### Problem: Responses are very slow

**Solutions:**
- Ensure only one model is loaded: `allama ps`
- Check GPU usage: `nvidia-smi`
- Reduce `max_model_len` if using long context
- Use a smaller/faster model variant (9b vs 35b)

---

## Advanced Configuration

### Custom Port

If Allama is running on a different port (not 9000):

1. Start Allama with custom port:
   ```bash
   ALLAMA_PORT=8888 allama serve
   ```

2. Configure OpenCode API URL:
   ```
   http://localhost:8888/v1
   ```

### Remote Allama (Different Machine)

1. Start Allama, make sure it binds to all interfaces:
   ```bash
   # Edit configs/server.py or set:
   ALLAMA_HOST=0.0.0.0 allama serve
   ```

2. In OpenCode, set API Base URL:
   ```
   http://192.168.1.100:9000/v1
   ```
   (Replace `192.168.1.100` with Allama's IP address)

### Multiple UIs (OpenCode + OpenWebUI)

You can run both simultaneously:

```bash
# Terminal 1: Start Allama
allama serve

# Terminal 2: Open OpenWebUI
allama ui openwebui

# Terminal 3: Open OpenCode
allama ui opencode
```

Both will connect to the same Allama server!

---

## Example Workflows

### Workflow 1: Quick Chat with Vision Model

```bash
# Terminal 1
allama serve

# Terminal 2
allama ui opencode

# In OpenCode:
# 1. Select "Qwen3vl:8b" from model dropdown
# 2. Upload an image
# 3. Ask questions about it (OCR, analysis, etc.)
```

### Workflow 2: Code Generation + Analysis

```bash
# Terminal 1
allama serve

# Terminal 2
allama ui opencode

# In OpenCode:
# 1. Select "Qwen3.5:35b-Code" for coding
# 2. Ask for code generation
# 3. Get code with explanations
# 4. Switch to "Qwen3.5:27b" for quick iterations
```

### Workflow 3: Compare Models

```bash
# Run two UIs side-by-side
allama serve    # Terminal 1
allama ui opencode   # Terminal 2
allama ui openwebui  # Terminal 3

# Same question in both UIs
# Compare outputs side-by-side
```

---

## API Compatibility

Allama exposes an **OpenAI-compatible API** at `http://localhost:9000/v1`:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions (streaming supported)
- `POST /v1/completions` - Text completions (optional)

This means any client that supports OpenAI API format will work with Allama, including:
- ✅ OpenCode
- ✅ OpenWebUI
- ✅ LiteLLM
- ✅ Custom scripts
- ✅ Third-party tools

---

## Comparison: OpenCode vs OpenWebUI

| Feature | OpenCode | OpenWebUI |
|---------|----------|-----------|
| **Web UI** | Modern, clean | Feature-rich |
| **Model Selection** | Dropdown | Search & select |
| **Image Upload** | ✅ | ✅ |
| **Model Switching** | Easy | Easy |
| **Chat History** | ✅ | ✅ |
| **Model Settings** | Basic | Advanced |
| **Performance** | Fast | Heavy |
| **Customization** | Limited | Extensive |

Both work equally well with Allama!

---

## Next Steps

- 📖 [Allama Configuration Guide](CONFIGURATION_ENCYCLOPEDIA.md)
- 🎯 [Your Setup Guide](YOUR_SETUP_GUIDE.md)
- 📊 [Model Benchmarks](YOUR_MODEL_CONFIG_REFERENCE.md)

---

**Last Updated:** April 2026
**Tested with:** OpenCode 1.4.0, Allama latest
