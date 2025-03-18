#!/usr/bin/env bash
unset http_proxy
unset https_proxy

port=12181

CUDA_VISIBLE_DEVICES=7 python llama_api.py \
    --port ${port} \
> "../logs/llama_${port}.log" 2>&1 &

# kill -9 $(lsof -t -i :12181)