conda create --name EfficientLM python=3.9
conda activate EfficientLM
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip3 install git+https://github.com/huggingface/accelerate
pip3 install -r requirements.txt
git clone https://github.com/Language-Modelling/submodlib.git
cd submodlib
pip3 install .
conda install -c conda-forge faiss-gpu
cd ..
conda deactivate