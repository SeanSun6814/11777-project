source /mnt/efs/fs1/mmml/miniconda3/etc/profile.d/conda.sh
conda activate cenv_x86
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=/mnt/efs/fs1/mmml/hf_home
export HUGGINGFACE_HUB_CACHE=/mnt/efs/fs1/mmml/hf_home
export HF_DATASETS_CACHE=/mnt/efs/fs1/mmml/hf_datasets_cache
