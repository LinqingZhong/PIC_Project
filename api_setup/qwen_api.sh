#!/usr/bin/env bash
unset http_proxy
unset https_proxy
port=12183

CUDA_VISIBLE_DEVICES=7 python qwen_api.py \
    --port ${port} \
> "../logs/qwen_${port}.log" 2>&1 &

# kill -9 $(lsof -t -i :12183)