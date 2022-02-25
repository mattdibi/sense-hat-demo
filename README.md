# sense-hat-demo
Kura AI Wire Component Sense Hat-based demo

## Instructions

For running these models inside Triton, navigate to this repository and run:

```bash
docker run --rm \
    -p8000:8000 \
    -p8001:8001 \
    -p8002:8002 \
    --shm-size=150m \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:22.01-py3 \
    tritonserver --model-repository=/models
```

Excpected models folder structure:

```bash
models
├── postprocessor
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
├── preprocessor
│   ├── 1
│   │   └── model.py
│   └── config.pbtxt
└── tf_autoencoder_fp32
    ├── 1
    │   └── model.savedmodel
    │       ├── assets
    │       ├── keras_metadata.pb
    │       ├── saved_model.pb
    │       └── variables
    │           ├── variables.data-00000-of-00001
    │           └── variables.index
    └── config.pbtxt
```
