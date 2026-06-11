import os
import sys
import json
import urllib.request
from urllib.error import HTTPError

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Generate a Render API Key from: https://dashboard.render.com/it/settings#api-keys
RENDER_API_KEY = "PASTE_YOUR_RENDER_API_KEY_HERE"

REPO_URL = "https://github.com/22ALR047/Driver-Drowsiness-Detection"
SERVICE_NAME = "driver-drowsiness-api"
# ==============================================================================

if RENDER_API_KEY == "PASTE_YOUR_RENDER_API_KEY_HERE":
    print("[ERROR] Please replace the RENDER_API_KEY placeholder with your actual Render API Key.")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {RENDER_API_KEY}",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# 1. Fetch Owner ID
print("[1/3] Retrieving Render owner ID...")
req = urllib.request.Request("https://api.render.com/v1/owners?limit=20", headers=headers)
try:
    with urllib.request.urlopen(req) as res:
        owners = json.loads(res.read().decode())
        if not owners:
            print("[ERROR] No owners found for this API Key.")
            sys.exit(1)
        # Select first owner account
        owner_id = owners[0]["owner"]["id"]
        print(f"[SUCCESS] Found Owner ID: {owner_id}")
except HTTPError as e:
    print(f"[ERROR] Failed to fetch owner ID: {e.read().decode()}")
    sys.exit(1)

# 2. Create Web Service on Render
print(f"[2/3] Creating Web Service '{SERVICE_NAME}' on Render...")
payload = {
    "type": "web_service",
    "name": SERVICE_NAME,
    "ownerId": owner_id,
    "repo": REPO_URL,
    "branch": "main",
    "rootDir": "Drowsiness_Detection",
    "autoDeploy": "yes",
    "serviceDetails": {
        "env": "python",
        "buildCommand": "pip install -r Drowsiness_Detection/requirements.txt",
        "startCommand": "python Drowsiness_Detection/backend/server.py",
        "plan": "free",
        "envVars": [
            {
                "key": "PYTHON_VERSION",
                "value": "3.10.12"
            }
        ]
    }
}

req = urllib.request.Request(
    "https://api.render.com/v1/services",
    data=json.dumps(payload).encode(),
    headers=headers,
    method="POST"
)

try:
    with urllib.request.urlopen(req) as res:
        service = json.loads(res.read().decode())
        service_id = service["id"]
        deploy_url = service["serviceDetails"]["url"]
        print(f"[SUCCESS] Service created! ID: {service_id}")
except HTTPError as e:
    response_body = e.read().decode()
    # Check if the service already exists
    if "already in use" in response_body:
        print("[INFO] Web Service already exists. Deploying existing service...")
        # Get services list to find existing ID
        req_list = urllib.request.Request("https://api.render.com/v1/services?limit=50", headers=headers)
        with urllib.request.urlopen(req_list) as res_list:
            services = json.loads(res_list.read().decode())
            match = next((s["service"] for s in services if s["service"]["name"] == SERVICE_NAME), None)
            if match:
                service_id = match["id"]
                deploy_url = match["serviceDetails"]["url"]
                print(f"[SUCCESS] Linked existing Service ID: {service_id}")
            else:
                print(f"[ERROR] Found name conflict but could not retrieve service ID: {response_body}")
                sys.exit(1)
    else:
        print(f"[ERROR] Failed to create service: {response_body}")
        sys.exit(1)

# 3. Trigger Deploy
print("[3/3] Triggering deployment deploy-hook...")
req_deploy = urllib.request.Request(
    f"https://api.render.com/v1/services/{service_id}/deploys",
    headers=headers,
    method="POST"
)
try:
    with urllib.request.urlopen(req_deploy) as res_deploy:
        deploy_info = json.loads(res_deploy.read().decode())
        print("\n==========================================================================")
        print("[SUCCESS] Render deployment started!")
        print("==========================================================================")
        print(f"Deploy ID: {deploy_info['id']}")
        print(f"Status   : {deploy_info['status']}")
        print(f"Public API URL: {deploy_url}")
        print("\nMonitor build logs on the Render Dashboard web UI.")
        print("Once the deployment is complete, paste the Public API URL into Vercel!")
        print("==========================================================================")
except HTTPError as e:
    print(f"[ERROR] Failed to trigger deploy: {e.read().decode()}")
    sys.exit(1)
