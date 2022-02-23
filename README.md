# sense-hat-demo
Kura AI Wire Component Sense Hat-based demo

## Instructions

Running the models inside triton

```bash
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 --shm-size=150m -v [path-to-this-repo]/models:/models tritonserver tritonserver --model-repository=/models
```
