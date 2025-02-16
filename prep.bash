# compile llama.cpp wit CUDA

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make clean && make LLAMA_CUDA=1

# more speed with CUDA, maybe

make LLAMA_CUDA=1 LLAMA_CUDA_FORCE_MATMUL=1

# decide for Mixtral or Mistral
# Mixtral (bigger context's)
wget https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main/mixtral-8x7b.Q4_K_M.gguf

# Mistral (lightweight, faster)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# start model - example for mistral:

# most important parameters:
# -m <pfad>	LLM-Modell (path to .gguf)
# -c <wert>	context length (Std: 512, for big queries: 4096)
# --n-gpu-layers <number>	model runnung at. (more = faster, but if too high, it ma crashes if you have not enought vram, 35 for 6G as rule of thumb)
# --temp <wert>	temp, of creativity (0.1 = deterministic, 1.0 = creative)
# --top-k <wert>	Limits the chosing of words to k most possible words (z.B. 40)
# --top-p <wert>	probability limit for words (z.B. 0.95)
# --batch-size <wert>	increases the interference rate by enable parallel execution

# ***** examples *****
# balanced:
./main -m mistral-7b-instruct-v0.1.Q4_K_M.gguf --n-gpu-layers 35 --threads 8 --batch-size 512 -c 4096

# faster. lower quylity:
./main -m mistral-7b-instruct-v0.1.Q4_K_M.gguf --n-gpu-layers 20 --ctx-size 2048 --top-k 40 --top-p 0.9 --temp 0.7

# debug/logging
./main -m mistral-7b-instruct-v0.1.Q4_K_M.gguf --n-gpu-layers 35 -c 4096 --verbose-prompt --log-disable

# optimized for 1060
./main -m mistral-7b-instruct-v0.1.Q4_K_M.gguf --n-gpu-layers 30 --threads 8 --ctx-size 2048 --batch-size 256 --top-k 40 --top-p 0.9 --temp 0.7

# top quality
# --n-gpu-layers 10	CPU means higher quality by decreased use of GPU
# --ctx-size 4096	long text and complex answers
# --batch-size 128	small batch size = more stability but slower
# --top-k 100	Model draws from the 100 most probable words (higher diversity)
# --top-p 0.99	Almost all possible words can be chosen (very broad thinking)
# --temp 0.2	Very deterministisch, nearly no random choosing = highest precision
# --threads 8	Max. CPU-Threads (if you have a fast cpu...)
./main -m mistral-7b-instruct-v0.1.Q4_K_M.gguf --n-gpu-layers 10 --ctx-size 4096 --batch-size 128 --top-k 100 --top-p 0.99 --temp 0.2 --threads 8

