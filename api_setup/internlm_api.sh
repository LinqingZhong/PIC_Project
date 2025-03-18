#!/usr/bin/env bash
unset http_proxy
unset https_proxy
port=12182

CUDA_VISIBLE_DEVICES=7 python internlm_api.py \
    --port ${port} \
> "../logs/internlm_${port}.log" 2>&1 &

# kill -9 $( lsof -t -i :12182)