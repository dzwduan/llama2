# LLaMA2.C 改进版



## 如何运行 

```bash
python3 export.py tinyllama-15M-V2.bin --version 2   --hf /nfs/home/xiaoxiao/models/hf_models/tinyllama-15M
python3 export.py tinyllama-1.1B_q80.bin --version 2 --hf /nfs/home/xiaoxiao/models/hf_models/TinyLlama-1.1B
make clean && make
./run tinyllama-15M-V2.bin -n 100
./run tinyllama-1.1B_q80.bin -n 100
```


## TODO
<!-- 1. 解耦bpe encode decode ， 不需要改 -->
1. 支持flash-attention
1. 支持 MLA
1. 支持CUDA单核运行