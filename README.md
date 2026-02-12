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

## Running the semantic search demonstrator

Once you have a model trained with masked language modelling (either from the `local_train.py` or federated job described above), 
you can run it through the semantic search pipeline. This depends on some files and is done in multiple steps.

The semantic search assumes that you have trained on some dataset of text,
containing examples of terms which are _positive_ and _negative_. Typically,
you'll use this large text to create a space of reference vectors which will
then be used for nearest neighbour classification of some new set of vectors (the query vectors used for evaluation).

### Files needed for evaluation

There are two main files needed for the evaluation: the _positive word_ lists and the _stop word_ list

#### Positive word files
To tag the vectors with a positive and negative class, we use keywords. For the evaluation, we'd like to use a different set of keywords for the reference vectors to the evaluation vectors to simulate there being relevant words we didn't know about. You should divide your set of keywords into three splits:
 1. **positive-words-train.txt**: A list of words used to tag the reference vectors. These will act as the neighbourhoods used to predict the class of query vectors.
 2. **positive-words-dev.txt**: A separate list of words to tag the ground truth of the query vectors of the development set. This will be used to tune the nearest neighbour algorithm (how many neighbours, threshold at which to classify as positive/negative, etc.). Evaluation criteria will be based on the agreement of words tagged with this list compared to the classes of the reference vectors. It's therefore important that these words are distinct from the list of words used to tag the reference.
 3. **positive-words.test.txt**: A separate list of words to tag the ground truth of the query vectors of the development set. This will be used to estimate final test performance. These should also be distinct from the training word list to get a good estimate of performance. It should also be distinct from the development words, but this is not as critical as the reference word classes will not have been tagged using the development list.

 Ideally, the words in these different lists should also differ in string similarity if you want to mainly evaluate semantic search performance.

#### Stop word file
While not strictly necessary, you will benefit from having a list of words to ignore. 
These are words you know beforehand are not relevant, but also dominate the frequency in the data (think of words like `a`, `of`, `and`, `the`, `or` etc. as well as punctuation marks). If you don't use a stop list, all these terms and their vectors will also be stored in the vector database, _vastly_ increasing the storage requirements while mostly hurting semantic search performance.


**N.b. the files can have whatever names you'd like, they are supplied as arguments to the relevant scripts**


### Selecting the model
If you've run the local training or federated training pipelines in particular, you will have multiple checkpoints. You could run the evaluation pipeline for each of the checkpoint and select the best one based on development set performance, but this is prohibitively costly in terms of compute. It is suggested that you instead select the best checkpoint based on the masked language modelling performance on your local development dataset. 

If you have run the local training script, this is not needed as you will already have a checkpoint for the best performing model, but if you have run federated training, you will likely instead have a set of models where the "best" checkpoint will be the best MLM performer of the whole federation. 

The script `aimplant_demonstrator/test_mlm_performance.py` will evaluate all checkpoints under a directory against your development data to give you the best performing model. Usage:

```bash
$ python aimplant_demonstrator/test_mlm_performance.py PATH_TO_EXPERIMENT_ROOT --model-filter "latest_model*.pt" [--dev-data PATH_TO_DEV_DATASET] [--test-data  PATH_TO_TEST_DATASET]
```

This will do a recursive search of all models fitting the argument to `--model-filter` from the root directory on the text files you optionally provide. If you don't provide a development and/or test dataset, it will use the ones you have specified in your `~/.federatedhealth/config.json` file (see above).

The results will be put in a directory under the experiment root called `local_test_results`. There will be a symlink in this directory to the best performing model with the path `local_test_results/best_local_model.ckpt`


### Preparing the data

Before running evaluation, you need to prepare a database with your reference vectors. This uses [LanceDB](https://lancedb.com/) for efficient storage and search of huge amounts of vectors. Create the database by running the following:

```bash
$ python aimplant_demonstrator/calculate_to_vector_database.py PATH_TO_MODEL_FILE PATH_TO_REFERENCE_DATA --stop-list PATH_TO_STOP_LIST --positive-words PATH_TO_POSITIVE_TRAIN_WORDS [--output-dir PATH_TO_WRITE_DATABASE_TO]
```

If you don't supply and output directory, it will be created under a subdirectory `vector_database` in the same directory as the model file. This will create a local `Lance` database with your vectors, one vector for _each_ word of the reference dataset. **This can be very demanding on storage**. Each word will be stored as a vector with 16 bit precision, and for the examples in this repository that means a 768 dimensional 16 bit floating point vector -- 1.5 kB per word.

### Creating query responses

Once the vector database has been constructed, we need to query the neighbourhoods of the evaluation vectors of the development set:

```bash
$ python aimplant_demonstrator/query_neighbourhoods.py PATH_TO_MODEL_FILE PATH_TO_DEV_DATASET PATH_TO_DATABASE_DIRECTORY --stop-list PATH_TO_STOP_LIST --target-positive PATH_TO_POSITIVE_DEV_WORDS --known-positive PATH_TO_POSITIVE_TRAIN_WORDS [--output-dir PATH_TO_STORE_NEIGHBOURHOOD_FILES]
```

Note that for the evaluation we want to focus on the simulated unknown but relevant words. The development set might also contain words which are in the positive training dataset, but we don't want to include them in the evaluation. Tagging them as negative (because they're not in the positive dev words list) would add lots of incorrect false positives (they are very likely to be close to words tagged positive in the reference), while adding them as additional positive words would greatly overestimate sensitivity. We will therefore need to specifically tell the evaluation pipeline that they are words to not include in evaluation, hence the `--known-positive` command line argument.

The neighbourhoods will by default be stored in a subdirectory next to the vector database if the `--output-dir` is omitted. The results are stored in numbered pickle files, containing a dictionary with two keys:
 - `"class_mapping"` : A dictionary which described the numerically encoded classes, and
 - `"neighbourhoods"`: A list of the query words, their label and the neighbours.

The neighbourhood lists have entries of the shape `("query_word", query_word_class), [(distance_neighbour_1, "word_neighbour_1", class_neighbour_1), (distance_neighbour_2, "word_neighbour_2", class_neighbour_2)]` where the neigbhours are sorted by distance (closest first).
Stop words and known positives will have empty neighbour lists, but are included so that the original sequence 
of words can be reconstructed.
Note that the search is done for each word in the development set, they are not aggregated into unique terms before the neighbourhood search to simulate the real application of searching over sequences of words.

### Determining hyper parameters

Once the neigbhourhoods have been computed, we need to determine what hyper parameters to use for actual search. This is done by comparing the query words of the dev set with the reference set (ignoring stop words and know positive words). The search is based on distance weighted nearest neighbours which is done in two steps:
 1. Select reference vector neighbourhood of size $k$.
 2. in $k$-size neighbourhood, weight the class of each refererence class by the distance of its vector using a weighting function $f(d)$.

The hyper parameters are $k$ and which $f$ to use.

Run the hyper parameter search:
```bash
$ python aimplant_demonstrator/analyze_neighbourhoods.py PATH_TO_NEIGHBOURHOODS_DIR [--output-dir PATH_TO_SAVE_ANALYSIS_RESULTS]
```

If `--output-dir` is omitted, it will be saved to a subdirectory called `analysis` of the neighbourhood files directory.

This will read all the neighbourhood files into compact memory format which should be manageble by most reasonable computers. The memory requirements should be roughly `number_of_dev_words*neighbourhood_size*5` bytes, so a development set of 1 million words with neighbourhoods size of 60 would be about 300 MB.

The evaluation will identify the top performing set of hyper parameters based on the Youden J-statistic (sensitivity + specificity - 1) on the development dataset. The results will be saved in a file called `analyzed_neighbourhoods.json` which contain the results as well as ROC AUC scores for the different hyper parameters. A plot of the same information will be created as `roc_auc_vs_neighbours.png` in the output directory.
