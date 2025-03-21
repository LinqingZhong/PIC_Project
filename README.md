<p align="center">
  <h1 align="center">A LLM-powered chat assistant for the energy industry</h1>
  <h3 align="center">
    <a href="1215788174@qq.com">Ting Cai</a>, <a href="cute030422@outlook.com">He Wang</a>, <a href="2471583488@qq.com">Zhuyun Yuan</a>, <a href="lqzhong@buaa.edu.cn">Linqing Zhong</a>, <a href="hyk1371782@163.com">Yakun Hou</a>, <a href="dohachbihi4@gmail.com">Doha Chbihi</a>
  </h3>
</p>

## Overview

Coming soon

## Installation
If you are unable to access the GitHub website, you can open the hosts file with Notepad at <span style="color:blue">`C:\Windows\System32\drivers\etc\hosts`</span> (for Android system) or at <span style="color:blue">`etc/hosts`</span> (for Linux/Mac). At the end of the file, add the following lines:  
```bash
20.205.243.166 github.com  
20.205.243.166 www.github.com  
20.205.243.166 api.github.com  
20.205.243.166 assets-cdn.github.com  
20.205.243.166 raw.githubusercontent.com  
20.205.243.166 gist.github.com  
```
After saving and closing the file, refresh the DNS cache by opening the terminal and running:

For Windows:
```bash
ipconfig /flushdns  
```
For Linux/Mac:
```bash
sudo systemd-resolve --flush-caches  
```
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
pip install -r requirements.txt
```
Then, download the model checkpoints using the command . Make sure that you have suffient hard drive space. (Ensure that the path to the downloaded model contains only English characters.)
```bash
python download.py
```

## Inference
We provide two ways to conduct LLMs' reasoning, i.e., online inference and API call.

### Online infernce
Change the graphics card number and python file to utilize different models.
```bash
cd inference
CUDA_VISIBLE_DEVICES=0
python llama_online.py
```

### API call
You should suspend the API service. This process may take ~30 seconds. Check whether the API service is successfully started in the corresponding log file in the "PIC_Project/log" folder.(Ensure that the server IP address of the computer has been modified.)
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