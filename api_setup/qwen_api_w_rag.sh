#!/usr/bin/env bash
unset http_proxy
unset https_proxy
port=12185

CUDA_VISIBLE_DEVICES=0 python qwen_api_w_rag.py \
    --port ${port} \
> "../logs/qwen_${port}.log" 2>&1 &

# kill -9 $(lsof -t -i :12185)