<p align="center">
  <h1 align="center">A LLM-powered chat assistant for the energy industry</h1>
  <h3 align="center">
    <a href="1215788174@qq.com">Ting Cai</a>, <a href="cute030422@outlook.com">He Wang</a>, <a href="2471583488@qq.com">Zhuyun Yuan</a>, <a href="lqzhong@buaa.edu.cn">Linqing Zhong</a>, <a href="hyk1371782@163.com">Yakun Hou</a>, <a href="dohachbihi4@gmail.com">Doha Chbihi</a>
  </h3>
</p>

## Overview

Coming soon

## Installation

Create the conda environment. We recommend using different conda environments to avoid conflicts, such as "internlm", "llama" and "qwen".
Here is an example.
```bash
conda_env_name=llama
conda create -n $conda_env_name python=3.10.0 -y
conda activate $conda_env_name
```
After setting up the environment, some packages are necessary to be installed. We provide a requirements.txt. Please be careful about the version of these packages.

```bash
cd PIC_Project
pip install requirements.txt
```
Then, download the model checkpoints using the command . Make sure that you have suffient hard drive space.
```bash
python download.py
```

## Inference
We provide two ways to conduct LLMs' reasoning, i.e., online inference and API call.

### Online infernce
Change the graphics card number and python file to utilize different models.
```bash
cd inference
CUDA_VISIBLE_DEVICES=0 python llama_online.py
```

### API call
You should suspend the API service. This process may take ~30 seconds. Check whether the API service is successfully started in the corresponding log file in the "PIC_Project/log" folder.
```bash
cd api_setup
bash llama_api.sh
```
Subsequently, use the conmand below to start your chat.
```bash
cd inference
python llama_offline.py
```

## License

PIC_Project is carried out by six students of ECPK in Beihang University. For questions, please contact [Linqing Zhong](lqzhong@buaa.edu.cn) or [Zhuyun Yuan](2471583488@qq.com).