# Masked Language Modeling for the Federated Health project

This repository contain code to train an XLMRoberta model using masked language modelling and LoRA fine tuning. The code is based on the [NLP-NER](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/nlp-ner) example, incorporating code from the Huggingface [run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py) scripts.

As example datasets, the code uses works of Jane Austen and Shakespeare.


## Installation

The suggested installation creates a virtual environment which you will use to run you federated learning client. All code run in the federated environment _must_ be pre-installed on the system, the federation will not allow arbitrary code to be executed on the nodes.

For convencience, the code for the experiment can be installed as a "development" package. This means that any changes to the code will automatically be reflected in the environment package (e.g. by doing a `git pull`, you don't have to remember to reinstall the package).

Start by cloning this repo:
```shell
$ git clone git@github.com:eryl/aimplant.git
$ cd aimplant
```

Now you can install either using `pip` (you must have python 3.10 installed system-wide) or `uv` (manages python version for you). `uv` is the recommended method.

### With `pip`

If you're using pip, you need python 3.10 installed on the system (later versions of python might cause issues with package dependencies). If you have trouble installing python 3.10 system-wide, it is suggested that you use the `uv` install method below.

Use `pip` to install the dependencies:

```shell
$ python3.10 -m venv .venv
$ source .venv/bin/activate
(federatedhealth)$ python -m pip install -U pip #Upgrade pip
(federatedhealth)$ python -m pip install -e .   # This installs this code
```

### With Astral `uv` (recommended)

Astral `uv` is a fast and capable python packaging tool. It conveniently installs full python environments for you, including different versions of python. Install it following [this guide](https://docs.astral.sh/uv/getting-started/installation).

Once `uv` is installed and added to your path you can run the following in the project directory:

```shell
$ uv sync  # creates .venv using the correct python environment
$ source .venv/bin/activate
(federatedhealth)$ uv pip install -e .
```

### XLM-RoBERTa

You will also need the model [from huggingface](https://huggingface.co/FacebookAI/xlm-roberta-base). Download the model from the project sharepoint (`WP2_health_data_space/T2.2_federated_infrastructure/fl_infrastructure/nvidia_flare/models/xlm-roberta-base.tar.gz`). Download this file and extract it to some directory (e.g. `models/xlm-roberta`).


## Dataset

The datasets are expected to be regular text files (UTF-8 encoded) with the training examples. In the aiMPLANT demonstrator, the files is organized with one line per patient, with the clinical notes for each patient concatenated sequentially according to date of the note. Do note that sequences of text will follow new-lines, so the context window for MLM will not include text spanning multiple lines. 


## Configuration

Configuration is based on a json configuration file. The default file can be found in `src/federatedhealth/default_config.json`. The experiment will look for this file in `$HOME/.federatedhealth/config.json`, and if not found will copy the default config there. You can do this manually by running:

```shell
$ cp src/federatedhealth/default_config.json $HOME/.federatedhealth/config.json
```

The config file (`$HOME/.federatedhealth/config.json`) in will look something like this:

```json
{
    "model_path": "/path/to/xlmroberta-dir",
    "data_config": {
        "training_data": "/path/to/training_data.txt",
        "dev_data": "/path/to/dev_data.txt",
        "test_data": "/path/to/test_data.txt"
    },
    "training_args": {
        "mlm_probability": 0.1,
        "optimization_batch_size": 32,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "max_train_steps": null,
        "num_train_epochs": 10,
        "lr_scheduler_type": "linear",
        "num_warmup_steps": 0,
        "checkpointing_steps": null,
        "aggregation_epochs": 1
    },
    "lora_config": {
        "task_type": "TOKEN_CLS", 
        "inference_mode": false, 
        "r": 8, 
        "lora_alpha": 8, 
        "lora_dropout": 0.1,
        "bias": "all"
    }
}
```

You need to change these values:
 - `"model_path"`: Point this to the directory you extracted the XLM-RoBERTa model to 
 - `"training_data"`: This should be the full path to your training text file
 - `"dev_data"`: This should be the full path to your development text file
 - `"test_data"`: This should be the full path to your test text file


### Configure local batch size
Due to differences in compute capacity, you might want to override the device batch size (number of samples which gradients are computed on at a time). You can change the configuration values:

 - `"training_args.per_device_train_batch_size"`
 - `"training_args.per_device_eval_batch_size"`


## Local training with sample data

First make sure you have the config installed:

```shell
$ cp src/federatedhealth/default_config.json $HOME/.federatedhealth/config.json
```

Assuming the model is in `models/xlmroberta`, we add this to the config file:

```shell
$ sed -i "s#\(\"model_path\": *\"\)[^\"]*\"#\1$PWD/models/xlm-roberta\"#" $HOME/.federatedhealth/config.json
```
And we can set the training data paths in the same way:

```shell
$ sed -i "s#\(\"training_data\": *\"\)[^\"]*\"#\1$PWD/fedhealth_mlm_data/site-1_train.txt\"#" $HOME/.federatedhealth/config.json
$ sed -i "s#\(\"dev_data\": *\"\)[^\"]*\"#\1$PWD/fedhealth_mlm_data/site-1_dev.txt\"#" $HOME/.federatedhealth/config.json
$ sed -i "s#\(\"test_data\": *\"\)[^\"]*\"#\1$PWD/fedhealth_mlm_data/site-1_test.txt\"#" $HOME/.federatedhealth/config.json
```

This makes sure the config file has valid entries for model and datasets. You can now run the local training:

```shell
(federatedhealth)$ python local_train.py
```


<!--
## Dockerized run (not updated since 2025-06-30)

You can instead run the client using docker. The `build/Dockerfile` containts the build recipie. The build definition assumes there are certain files needed to be copied into the container. You need to download the XLM-RoBERTa model and place it in `models/xlm-roberta`.

Use the following to build the container:

```
$ cd build
$ docker build -t fedhealth .
```

To run the image, you need to mount in certain directories (path is inside container, the mount target):
 - `/app/data` - this directory should contain three text files, `training_data.txt`, `dev_data.txt` and `test_data.txt`.
 - `/app/client_kit` - this directory should contain your nvidia flare client directory, i.e. your credentials and the startup script.
 - `/app/workspace` - this is where all persistent data from the run will be saved (is this needed, if it's stored under client_kit that's probably for the best)

 To run the container, use the following command (replace the `/path/to/your/[...]` with the appropriate path):

 ```
 docker run --gpus all -it \
    -v /path/to/your/dataset:/app/data \
    -v /path/to/your/client_kit:/app/client_kit \
    fedhealth bash
 ```

This (running with `bash` as an argument) will start a shell inside the container which you can experiment with. There are two other entrypoints available: `simulator` and `client`. These are shortcuts to either start a simulated run inside the container (to test that things work) or to start in NVFLARE client mode, i.e. connect your computer to the federated network.

To run the simulator, you can for example run:

```
$ docker run --gpus all -it -v ./fedhealth_mlm_data:/app/data fedhealth simulate
```

-->