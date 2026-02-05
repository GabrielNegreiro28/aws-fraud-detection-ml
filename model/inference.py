import json
import os
import joblib
import numpy as np

def model_fn(model_dir: str):
    # O SageMaker extrai model.tar.gz em model_dir
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        payload = json.loads(request_body)

        # Espera {"features": [...]} ou {"instances": [[...], [...]]}
        if "features" in payload:
            X = np.array(payload["features"], dtype=float).reshape(1, -1)
            return X
        if "instances" in payload:
            X = np.array(payload["instances"], dtype=float)
            return X

        raise ValueError("JSON inválido. Use {'features':[...]} ou {'instances':[[...], ...]}")
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    # Retorna probabilidade de fraude (classe 1)
    proba = model.predict_proba(input_data)[:, 1]
    return proba

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps({"fraud_probability": prediction.tolist()}), accept
    return str(prediction), accept
