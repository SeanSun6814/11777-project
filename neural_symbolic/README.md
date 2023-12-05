# Generative Evaluation
Notebooks to perform generative evaluations of VLMS on winoground
You must execute them in this directory
```
AMR.ipynb
Caption.ipynb
Eval.ipynb
smatch/ # eval scripts for smatch score
```

## Environment setup
### AMR Parser Environment

```
git clone https://github.com/IBM/transition-amr-parser
cd transition-amr-parser
```
Create a `set_environment.sh` under `transition-amr-parser`.
You should modify them according to your setup. Here is our setup on AWS. 
The environments, models, datasets are cached on EFS to save costs, 
and this allows us to attach different compute instances to it (tested on Deep Learning Base AMI)
```
source /mnt/efs/fs1/mmml/miniconda3/etc/profile.d/conda.sh
conda activate cenv_x86
export CUDA_HOME=/usr/local/cuda-11.7
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export TRANSFORMERS_CACHE=/mnt/efs/fs1/mmml/hf_home
export HUGGINGFACE_HUB_CACHE=/mnt/efs/fs1/mmml/hf_home
export HF_DATASETS_CACHE=/mnt/efs/fs1/mmml/hf_datasets_cache
```
Install Environment
```
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
conda env create -f parser_env.yaml
```
If the conda create fails, you'd have to do it manually.
```
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
conda create -y -n cenv_x86 python=3.8
pip install transition-neural-parser
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install jupyterlab
pip install ipykernel
pip install transformers datasets accelerate bitsandbytes matplotlib
pip install dotenv
```
Put your huggingface token in a `.env` file

make this available on jupyter lab
```
conda activate cenv_x86
python -m ipykernel install --user --name cenv_x86 --display-name amr
```
### Fromage Environment
```
git clone https://github.com/kohjingyu/fromage.git
conda create -n fromage -y python=3.8
cd fromage
pip install -r requirements.txt
```
make this available on jupyter lab
```
conda activate fromage
python -m ipykernel install --user --name fromage --display-name fm
```

## Workflow
Launch Jupyter Lab
```
# you must source the set_environment.sh so that cuda points to 11.7
source transition-amr-parser/set_environment.sh && jupyter lab --ip "0.0.0.0" --no-browser
```
1. Generate AMR for reference caption: `AMR.ipynb`
2. Generate candidate captions using VLMS: `Captioning.ipynb` (`Fromage` notebook is WIP)
3. Parse the candidate captions: go back to `AMR.ipynb`, the relevant section is `Parse Model Predictions`
4. Evaluate amrs of generated caption against amrs of reference caption: `Eval.ipynb`



