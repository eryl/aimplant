# Masked Language Modeling for the Federated Health project

This repository contain code to train an XLMRoberta model using masked language modelling and LoRA fine tuning. The code is based on the [NLP-NER](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/nlp-ner) example, incorporating code from the Huggingface [run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm_no_trainer.py) scripts.

As example datasets, the code uses works of Jane Austen and Shakespeare.


## Installation

Use `pip` to install the dependencies:

```shell
$ python3 -m venv fh-mlm-env
$ source fh-mlm-env/bin/activate
(fh-mlm-env)$ python -m pip install -U pip #Upgrade pip
(fh-mlm-env)$ python -m pip install -r requirements.txt
```

### XLM-RoBERTa

You will also need the model [from huggingface](https://huggingface.co/FacebookAI/xlm-roberta-base). If you're computer has access to the internet you can just run the `download_model.py` script.

## Dataset

The datasets are expected to be regular text files (UTF-8 encoded) with the training examples. In the aiMPLANT demonstrator, the files is organized with one line per patient, with the clinical notes for each patient concatenated sequentially according to date of the note. Do note that sequences of text will follow new-lines, so the context window for MLM will not include text spanning multiple lines. 

## Configuration

To configure the training we will use environment variables. The reason for this instead of using e.g. command line arguments is that during the federated training, initialization of the tasks will be done by the server, so site-specific configurations shouldn't be handed out together with the task (where each site stores their data should be private information). An alternative could be to use a local configuration file, but we opt to use environment variables instead so that we don't have to rely on a file being at some hard coded path.

The following environmental variables need to be set:

 - `FH_MODEL_DIR`: The absolute path to the directory containing the XLM-RoBERTa model from huggingface.
 - `FH_TRAINING_DATA`: The absolute path to the file containing the training data
 - `FH_DEV_DATA`: The absolute path to the file containing the development (sometimes referred to as "validation" [sic] data) data
 - `FH_TEST_DATA`: The absolute path to the file containing the test data. The test data is not used by the federated learning task, but is included for completeness. If you don't have test data available, you can set this to the same path as the dev dataset.

Apart from this, during local training and simulation you might need to add the `apps/xlmroberta_mlm` directory to your `PYTHONPATH` environmental variable so that the `custom` package can be found.

```shell
$ export PYTHONPATH=$PWD/apps/xlmroberta_mlm:$PYTHONPATH  # Assuming that $PWD is the project root, this adds the app directory so that the custom code can be found
```

### Configure local batch size
Due to differences in compute capacity, you might want to override the device batch size (number of samples which gradients are computed on at a time). Two environmental variables control this and overrides the setting in the training configuration:

 - `FH_TRAIN_BATCH_SIZE` - size of batches of training samples. The number of gradient accumulation steps will automatically be set so that the effective batch size for optimization remains the same
 - `FH_EVAL_BATCH_SIZE`- size of batches for evaluation


## Example with sample data
```shell
export PYTHONPATH=$PWD/apps/xlmroberta_mlm:$PYTHONPATH  # Assuming that $PWD is the project root, this adds the app directory so that the custom code can be found
export FH_MODEL_DIR=$PWD/models/xlmroberta
export FH_TRAINING_DATA=$PWD/fedhealth_mlm_data/site-1_train.txt
export FH_DEV_DATA=$PWD/fedhealth_mlm_data/site-1_dev.txt
export FH_TEST_DATA=$PWD/fedhealth_mlm_data/site-1_test.txt
python local_train.py
```

## Dockerized run

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