import boto3

region = "us-east-1"
sm = boto3.client("sagemaker", region_name=region)

endpoint_name = "fraud-lr-endpoint-1770256320"

# Descobrir config e model associados
desc = sm.describe_endpoint(EndpointName=endpoint_name)
config_name = desc["EndpointConfigName"]

cfg = sm.describe_endpoint_config(EndpointConfigName=config_name)
model_name = cfg["ProductionVariants"][0]["ModelName"]

print("Endpoint:", endpoint_name)
print("EndpointConfig:", config_name)
print("Model:", model_name)

# Deletar na ordem correta
print("Deleting endpoint...")
sm.delete_endpoint(EndpointName=endpoint_name)

print("Deleting endpoint config...")
sm.delete_endpoint_config(EndpointConfigName=config_name)

print("Deleting model...")
sm.delete_model(ModelName=model_name)

print("✅ Cleanup started")
