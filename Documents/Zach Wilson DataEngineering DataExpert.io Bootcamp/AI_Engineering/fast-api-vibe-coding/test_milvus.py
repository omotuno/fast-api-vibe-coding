#!/usr/bin/env python3
"""Test script to diagnose Milvus connection issues"""

import os
from dotenv import load_dotenv
from pymilvus import MilvusClient

# Load environment variables
load_dotenv()

# Get credentials
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

print("=" * 60)
print("Milvus Connection Test")
print("=" * 60)
print(f"MILVUS_URI: {MILVUS_URI}")
print(f"MILVUS_TOKEN: {'*' * 20}...{MILVUS_TOKEN[-10:] if MILVUS_TOKEN else 'NOT SET'}")
print()

if not MILVUS_URI:
    print("❌ ERROR: MILVUS_URI is not set in .env file")
    exit(1)

if not MILVUS_TOKEN:
    print("❌ ERROR: MILVUS_TOKEN is not set in .env file")
    exit(1)

print("Attempting to connect...")
try:
    # Try connecting with MilvusClient
    client = MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN
    )
    print("✅ MilvusClient created successfully")
    
    # Test connection by listing collections
    print("\nTesting connection by listing collections...")
    collections = client.list_collections()
    print(f"✅ Connection successful! Found {len(collections)} collections:")
    for col in collections:
        print(f"   - {col}")
    
    print("\n✅ Milvus is connected and working!")
    
except Exception as e:
    print(f"\n❌ Connection failed!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Troubleshooting Tips:")
    print("=" * 60)
    print("1. Verify your MILVUS_TOKEN is correct in Zilliz Cloud console")
    print("2. Check if your Zilliz Cloud cluster is active")
    print("3. Verify the MILVUS_URI matches your cluster endpoint")
    print("4. Check your network connection")
    print("5. Try updating pymilvus: pip install --upgrade pymilvus")

