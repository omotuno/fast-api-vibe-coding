# How to Connect to Milvus (Zilliz Cloud)

## Prerequisites

1. **Zilliz Cloud Account** - You need a Zilliz Cloud account with an active cluster
2. **Milvus URI** - Your cluster endpoint URL
3. **Milvus Token** - Your API token from Zilliz Cloud

## Steps to Connect

### 1. Get Your Milvus Credentials from Zilliz Cloud

1. Log in to [Zilliz Cloud Console](https://cloud.zilliz.com)
2. Go to your cluster/project
3. Find your **Endpoint URI** (looks like: `https://xxx.serverless.gcp-us-west1.cloud.zilliz.com`)
4. Generate or copy your **API Token**

### 2. Update Your `.env` File

Make sure your `.env` file contains:

```bash
MILVUS_URI=https://your-cluster-endpoint.zilliz.com
MILVUS_TOKEN=your-api-token-here
```

### 3. Verify Your Setup

Run the test script:
```bash
python3 test_milvus.py
```

Or test directly in Python:
```python
from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN")
)

# Test connection
collections = client.list_collections()
print(f"Connected! Collections: {collections}")
```

### 4. Common Issues

#### Issue: "ModuleNotFoundError: No module named 'pymilvus'"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

#### Issue: "Connection failed" or "Authentication failed"
**Solutions:**
- Verify your token is correct in Zilliz Cloud console
- Check if your cluster is active (not paused)
- Ensure the URI matches your cluster endpoint exactly
- Try regenerating your API token

#### Issue: "Network error" or "Timeout"
**Solutions:**
- Check your internet connection
- Verify firewall isn't blocking the connection
- Try from a different network

### 5. Verify Connection in Your App

1. Start your server: `python main.py`
2. Check the console output - you should see:
   ```
   Attempting to connect to Milvus at: https://...
   Connected to Milvus successfully. Found X collections.
   ```
3. Visit `http://localhost:8000/rag-status` to check status via API
4. The UI should show "Active (Milvus Connected)" in the sidebar

### 6. If Still Not Working

Check the server console logs when it starts. Look for:
- Connection attempt messages
- Error messages with details
- Any warnings about missing credentials

The connection happens during server startup in the `lifespan` function.

