# LM-pretraining
## Efficient Pretraining of Language Models
### Environment Setup
#### Run the following in a sequence to set up the environment for running the code. (It is assumed that you have anaconda installed)
>- `conda create --name ingenious python=3.9`
>- `conda activate ingenious`
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

### Running the Code
Change appropriate parameters in `train_BERT.py` and run it. 

#### There are different scripts that execute different algorithms
- `run_language_modeling.py` runs traditional BERT pretraining, exactly similar to the Google's BERT paper
- `run_lm_with_subsets.py` runs the following algorithm: After every `select_every` steps, embeddings are computed for the full dataset, Using the submodular function mentioned as `selection_strategy`, a new subset is selected and this is used to train the model for next `select_every` steps. This process continues till `max_train_steps` training steps are completed
- `run_lm_with_subsets_knn.py` runs the following algorithm: An initial warmstart is done on the ground set. Assuming that the embeddings computed are not that bad to compute similarities and hence to measure redundancies, we use the computed embeddings at this point to select a subset using the submodular function mentioned as `selection_strategy`. After every 