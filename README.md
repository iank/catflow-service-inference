# catflow-service-inference

Inference service for an object recognition pipeline

# Setup

* Install [pre-commit](https://pre-commit.com/#install) in your virtualenv. Run
`pre-commit install` after cloning this repository.

# Develop

```
pip install --editable .[dev]
```

# Build

```
pip install build
python -m build
docker build -t iank1/catflow_service_inference:latest .
```

# Test

```
export CATFLOW_MODEL_NAME=your_model.pt
export CATFLOW_MODEL_THRESHOLD=0.5
pytest
```

Or `pytest --log-cli-level=DEBUG` for more diagnostics

# Deploy

See [catflow-docker](https://github.com/iank/catflow-docker) for `docker-compose.yml`
