import time
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

MODEL_DATA = "s3://gabriel-fraud-ml-portfolio/artifacts/training-jobs/sagemaker-scikit-learn-2026-02-05-01-19-10-420/output/model.tar.gz"

sess = sagemaker.Session()
role = sagemaker.get_execution_role()

# Nome único para evitar conflito
ts = int(time.time())
endpoint_name = f"fraud-lr-endpoint-{ts}"

print("Role:", role)
print("Model data:", MODEL_DATA)
print("Endpoint name:", endpoint_name)

# Usa o mesmo runtime do sklearn que treinou
model = SKLearnModel(
    model_data=MODEL_DATA,
    role=role,
    entry_point="inference.py",
    source_dir="model",
    framework_version="1.2-1",
    py_version="py3",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=endpoint_name,
)

print("✅ Endpoint created:", endpoint_name)
