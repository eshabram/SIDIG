# The Space Is Dirty Image Generator (SIDIG)


## Getting Started
It is highly recommended to use a virtual environment. If you use conda, it is recommended to use pip for package management. To install the dependencies, run this command from inside you virtual environment:
```shell
pip install -r requirements.txt
```

### Installing SDXL-Turbo
1. Download the SDXL Turbo weights from Hugging Face: https://huggingface.co/stabilityai/sdxl-turbo (log in and accept the license if prompted).

2. Authenticate so you can fetch gated assets:
   ```shell
   huggingface-cli login
   ```
3. Pull the model files into your local `models/` directory:
   ```shell
   huggingface-cli download stabilityai/sdxl-turbo --local-dir models/sdxl-turbo
   ```
4. Reference that directory (for example `--model_path models/sdxl-turbo`) when running fine-tuning or inference scripts.

### Run the Fine Tune
To tune the model, run this command, pointing to the model you would like to tune:
```shell
python sidig-fine-tuner.py
```

### Inference
For inference, we have designed a custom shell. This allows us to control the prompt header, adding in out token to make inference stronger, or instructions to try and prompt engineer the model to our liking. To run the prompt, simply run this script 