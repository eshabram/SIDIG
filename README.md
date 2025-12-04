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
To run the evaluation:
```shell
python evaluation.py
```
