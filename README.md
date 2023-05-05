# LM-pretraining
## Efficient Pretraining of Language Models
### Environment Setup
#### Run the following in a sequence to set up the environment for running the code. (It is assumed that you have anaconda installed)
>- `conda create --name EfficientLM python=3.9`
>- `conda activate EfficientLM`
>- `conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge`
>- `pip3 install git+https://github.com/huggingface/accelerate`
>- `pip3 install -r requirements.txt`
>- `git clone https://github.com/Language-Modelling/submodlib.git`
>- `cd submodlib`
>- `pip3 install .`
>- `conda install -c conda-forge faiss-gpu`
>- `cd ..`
>- `conda deactivate`
#### Configuring the accelerate library according to the training environment
Run `accelerate config` and answer the following questions
An example is given below
- In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): **0**
- Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): **2**
- How many different machines will you use (use more than 1 for multi-node training)? [1]: **1**
- Do you want to use DeepSpeed? [yes/NO]: **NO**
- How many processes in total will you use? [1]: **4**
- Do you wish to use FP16 (mixed precision)? [yes/NO]: **yes**

## On the Bookcorpus + English Wikipedia dataset
Prepare the dataset by running python3 utils/prepare_bookcorpus_wiki.py

### Running the Code
Change appropriate parameters in `train_BERT.py` and run it. 

Train BERT from scratch for 1,000,000 steps by running python3 train_BERT.py

#### There are different scripts that execute different algorithms
- `BERT\run_mlm_nsp.py` runs traditional BERT pretraining, exactly similar to the Google's BERT paper.
- `BERT\run_mlm_with_subsets_importance_sampling.py` runs the BERT pre-training on the subsets selected by our approach.
- `BERT\run_mlm_with_uncertatinty_sampling.py` runs the loss based sampling baseline.
- `BERT\run_mlm_subsets_fixed_subset.py` runs the the random subset selection baseline.

## On the OpenWebText dataset
Prepare the dataset by running python3 utils/prepare_gpt2_corpus.py

### Running the Code
Change appropriate parameters in `train_gpt2.py` and run it. 


#### There are different scripts that execute different algorithms
- `GPT2\run_clm.py` runs traditional GPT-2 pretraining, exactly similar to the OpenAI GPT-2 paper.
- `GPT2\run_clm_with_subsets_importance_sampling.py` runs the GPT-2 pre-training on the subsets selected by our approach.
