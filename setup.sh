
#!/bin/bash

pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

pip install -r requirements.txt