mkdir -p scripts
cat > scripts/invoke_endpoint.py << 'EOF'
import json
import argparse
import boto3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()

    smr = boto3.client("sagemaker-runtime", region_name=args.region)
    payload = {"features": [0]*30}

    resp = smr.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload).encode("utf-8"),
    )
    print(resp["Body"].read().decode("utf-8"))

if __name__ == "__main__":
    main()
EOF

