#!/usr/bin/env bash
unset http_proxy
unset https_proxy
port=12186

CUDA_VISIBLE_DEVICES=0 python internlm2_5_api_sft.py \
    --port ${port} \
> "../logs/internlm_${port}.log" 2>&1 &

# kill -9 $( lsof -t -i :12186)