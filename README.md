# Serverless | Model Checkpoint Template

End-to-End template for deploying your own Stable Diffusion Model to RunPod Serverless.

The setup scripts will help to download the model as well as setting up the Dockerfile.

## Setup

```BASH
git clone https://github.com/runpod/serverless-ckpt-template.git

docker build --build-arg MODEL_URL={hf.co/model/id} -t repo/serverless-ckpt-template .
```
