# The Space Is Dirty Image Generator (SIDIG)


## Getting Started
It is highly recommended to use a virtual environment. To install the dependencies, run this command from inside you virtual environment:
```shell
pip install -r requirements.txt
```
OR if you're really slick, you can use Conda:
```
conda create -n sidig python=3.12 -y
conda activate sidig
pip install -r requirements1.txt
pip install -r requirements2.txt
```
Note: files in requirements1.txt are from a specific index, so they are separated for clarity and specificity

### Installing SDXL-Turbo
When running the fine tune or prompt script, the sdxl-turbo model will be automatically downloaded and take up about 7gb.

### Run the Fine Tune
To tune the model, run this command, pointing to the model you would like to tune:
```shell
python sidig-fine-tuner.py
```

### Inference
For inference, we have designed a custom shell. This allows us to control the prompt header, adding in out token to make inference stronger, or instructions to try and prompt engineer the model to our liking. To run the prompt, simply run this script 

### Evaluation
The evaluation is split into two parts, a script to generate the images, and another script to use the generated images.

The images can be generated using the following python script:
```shell
python generate_evaluation_dataset.py
```
The numpy lists are used for CLIP score, and the .png images are used for VQAScore.

To run the evaluation, t2v-metrics is used to calculate the VQAScore, which requires a separate environment with dependencies that would conflict with our own.

After creating an environment with:
```shell
conda create -n vqa python=3.10 -y
conda activate vqa
```

Follow the instructions to install t2v-metrics, listed [at the t2v_metrics repo:](https://github.com/linzhiqiu/t2v_metrics?tab=readme-ov-file#quick-start)

```shell
conda install ffmpeg -c conda-forge
pip install t2v-metrics

# Install Git-based dependencies
pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/linzhiqiu/pytorchvideo.git

# Install flash-attention (CUDA 12.2, Python 3.10)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

The final step is to install torchmetrics
```shell
pip install torchmetrics
```

Once the evaluation environment (vqa) is set up, the evaluation metrics can be reproduced with:
```shell
python evaluation.py
```
