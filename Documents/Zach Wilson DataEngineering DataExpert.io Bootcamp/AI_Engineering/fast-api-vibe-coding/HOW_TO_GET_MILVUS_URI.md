# How to Get Your Milvus URI from Zilliz Cloud

## Steps to Find Your Milvus URI

### Method 1: Zilliz Cloud Console (Recommended)

1. **Log in to Zilliz Cloud**
   - Go to https://cloud.zilliz.com
   - Sign in with your account

2. **Navigate to Your Cluster**
   - Click on your project/cluster from the dashboard
   - Or go to "Clusters" in the left sidebar

3. **Find the Endpoint/URI**
   - Look for a section called:
     - **"Endpoint"** or
     - **"Connection URI"** or
     - **"Public Endpoint"**
   - It will look something like:
     ```
     https://in03-xxxxx.serverless.gcp-us-west1.cloud.zilliz.com
     ```
   - Or for other regions:
     ```
     https://xxx.aws-us-east-1.vectordb.zillizcloud.com
     ```

4. **Copy the Full URI**
   - Copy the entire URL (including `https://`)
   - This is your `MILVUS_URI`

### Method 2: Connection Details Page

1. In your cluster page, look for:
   - **"Connect"** button
   - **"Connection Details"** section
   - **"API Endpoint"** field

2. The URI is usually displayed in connection examples or API documentation

### Method 3: Check Your Cluster Settings

1. Go to **Cluster Settings** or **Configuration**
2. Look for **"Public Endpoint"** or **"API Endpoint"**
3. Copy that URL

## Update Your .env File

Once you have your URI, update your `.env` file:

```bash
MILVUS_URI=https://your-actual-endpoint.zilliz.com
MILVUS_TOKEN=3f51844f1bf212ee872c7530bc19e758b2e029001686b24322c2b4e901ca333577637c65ea9dbbf5bb2d0e5ddc5dda72286dbbb4
```

## What the URI Should Look Like

- **Serverless (GCP)**: `https://in03-xxxxx.serverless.gcp-us-west1.cloud.zilliz.com`
- **Serverless (AWS)**: `https://xxx.aws-us-east-1.vectordb.zillizcloud.com`
- **Dedicated**: `https://xxx.gcp-us-west1.vectordb.zillizcloud.com`

## Important Notes

- The URI is **different** from your cluster name
- It's the **public endpoint** for API connections
- Make sure you copy the **full URL** including `https://`
- The URI is **region-specific** (e.g., `gcp-us-west1`, `aws-us-east-1`)

## If You Can't Find It

1. Check the Zilliz Cloud documentation
2. Look in the "Quick Start" or "Getting Started" section
3. Check your email for cluster creation confirmation (it might contain the endpoint)
4. Contact Zilliz Cloud support

