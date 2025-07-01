# LLaMA2.C 改进版

## 软件层面TODO

1. 支持运行tinyLlama-1.1B
1. GQA支持
1. pd分离，同时用上gemm + gemv
1. 添加awq量化支持并解耦当前的量化实现
1. BF16支持


## 如何运行 

```bash
python3 export.py tinyllama-15M-V2.bin --version 2   --hf /nfs/home/xiaoxiao/models/hf_models/tinyllama-15M
python3 export.py tinyllama-1.1B_q80.bin --version 2 --hf /nfs/home/xiaoxiao/models/hf_models/TinyLlama-1.1B
make clean && make
./run tinyllama-15M-V2.bin -n 100
./run tinyllama-1.1B_q80.bin -n 100
```