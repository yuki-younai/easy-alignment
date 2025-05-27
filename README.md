

1. python环境
```c
conda create -n openr1 python=3.11.0

cd ~
cd .cache/

```
2. torch
```
# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
3. package
```c
pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install datasets  -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install trl==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install vllm==0.7.3 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
4. flash attention
```c
https://github.com/Dao-AILab/flash-attention/releases
下载: 
[flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl)
```