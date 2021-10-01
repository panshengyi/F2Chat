# F2Chat

## Project Description
This project contains the code and dataset for our ASE 2021 paper `Automating Developer Chat Mining`.

We conduct an in-depth analysis to identify the potential information categories of developer discussion threads that may satisfy the needs of various Open Source Software (OSS) stakeholders. We build a hierarchical taxonomy with nine information categories:

1. Problem Report
   * Programming Problem
   * Library Problem
   * Documentation Problem
2. Information Retrieval
   * Programming Information
   * Library Information
   * Documentation Information
   * General Information
3. Project Management
   * Technical Discussion
   * Task Progress

You can refer to the paper for definitions and examples.

We collect chat data from three active chatrooms on Gitter: [Angular](https://gitter.im/angular/angular), [Spring-boot](https://gitter.im/spring-projects/spring-boot) and [Deeplearning4j](https://gitter.im/eclipse/deeplearning4j). We preprocess the raw chatdata (e.g., thread disentanglement). **We build a dataset with 2,959 discussion threads and labelled using the defined taxonomy.** The annotation process cost over **480 person hours**. The dataset is in [angular.json](./F2Chat/data/angular.json), [spring-boot.json](./F2Chat/data/spring-boot.json) and [deeplearning4j.json](./F2Chat/data/deeplearning4j.json).

**We further propose a novel classification approach**, namely **F2Chat**, which combines handcrafted non-textual **F**eatures with deep textual **F**eatures extracted by neural models. Specifically, F2Chat has two stages:

* **Stage one**: Pretrain the **T**extual feature encoder using the siamese architecuture (F2Chat-t)
* **Stage two**: Facilitate in-depth fusion of both textual and non-textual features.


## Environments

1. OS: Ubuntu

   Memory: minimum 32G

   GPU: minimum 16G, e.g., NVIDIA Tesla T4. optimum 32G, e.g., NVIDIA Tesla V100.

2. Language: Python (v3.8)

3. CUDA: 10.2

4. Python packages:
   * [PyTorch 1.7.0](https://pytorch.org/)
   * [AllenNLP 1.3.0](https://allennlp.org/)
   * [Transformers 4.0.1](https://huggingface.co/)
   
   Please refer the official docs for the use of these packages (especially **AllenNLP**).

5. Setup:

   We use the approach proposed by Shi *et al.* ([Detection of Hidden Feature Requests from Massive Chat Messages via Deep Siamese Network, ICSE 2020](https://ieeexplore.ieee.org/abstract/document/9283914)), named **FRMiner**, as our baseline. The artifacts of their work are archived at [link](https://archive.softwareheritage.org/browse/origin/https://github.com/FRMiner/FRMiner/directory/). However, their task is slightly defferent from ours. We adapt their code to our task, and you can find the adapted code [here](./Baseline/).

   Please download [Glove](http://nlp.stanford.edu/data/glove.6B.zip), then unzip this file and put `glove.6B.50d.txt` into `./F2Chat/data` folder.

   We use [BERT-Small](https://arxiv.org/abs/1908.08962) from HuggingFaces Transformer Libarary ([link](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4)). You don't need to download the pretrained model by yourself as it will be downloaded the first time you run the code.

## Dataset

**Dataset has 2,959 discussion threads from three chatrooms on Gitter in total.** The annotation process cost over **480** person hours. The dataset is in [angular.json](./F2Chat/data/angular.json), [spring-boot.json](./F2Chat/data/spring-boot.json) and [deeplearning4j.json](./F2Chat/data/deeplearning4j.json). Follow is a sample:

```
{
    "id": 7,
    "project": "angular",
    "intention": "documentation_information",
    "messages": [
        {
            "id": "5cf781f782c2dc79a553bbf3",
            "text": "This article looks to be still updated https://netbasal.com/when-to-unsubscribe-in-angular-d61c6b21bad3 ? ",
            "time": "2019-06-05 08:48:55",
            "index": 0,
            "user": "xavadu_twitter"
        },
        {
            "id": "5cf782da702b7e5e76312951",
            "text": "@xavadu_twitter it seems still up to date to me.",
            "time": "2019-06-05 08:52:42",
            "index": 1,
            "user": "jocelynlecomte"
        },
        {
            "id": "5cf782e7e41fe15e7515db10",
            "text": "thank you @jocelynlecomte ",
            "time": "2019-06-05 08:52:55",
            "index": 2,
            "user": "xavadu_twitter"
        }
    ]
},
```

The meaning of the attributes are:

* *id*: threads in the same chatroom have unique id. threads in different chatrooms may have the same id
* *project*: the chatroom where the thread is from
* *intention*: labelled information types according to our defined taxonomy
* *messages*: messages in the discussion thread (already disentangled).
   * *id*: id prvided by Gitter API. Each message has unique id
   * *text*: plain text
   * *time*: sent time
   * *index*: relative position w.r.t. the first message of the thread
   * *user*: name of the user who sent the message

Message attributes except *index* are provided by [Gitter API](https://developer.gitter.im/docs/messages-resource). You can read the official docs for more details. The attribute of *index* added after thread disentanglement for support of certain non-textual features.

## File Organization
There are two files (for testing) and three directoris (`Baseline` - FRMiner, `F2Chat` - our proposed two-stage model, `F2Chat-s` - ablation study).

### Files

* `predict.py`: testing of Baseline, F2Chat-t (stage one) and F2Chat-s (ablation study).

* `predict_stage2.py`: testing of F2Chat (stage two).

### Directories

* `Baseline/`: code of FRMiner. We adapt it to our task and data. The changes are explained in detail through code comments.

   * `reader.py`: dataset reader for FRMiner
   * `model.py`: FRMiner model
   * `siamese_metric.py`: validation metric for training siamese network
   * `util.py`: util fuctions, e.g., replacement of special tokens, generation of train set and test set.
   * `config.json`: a json file including settings. Please refer to official docs of [AllenNLP](https://allennlp.org/) for more details.

* `F2Chat/`: code of F2Chat-t (stage one) and F2Chat (stage two). Datasets used in the experiments.

   * `reader.py`: dataset reader for F2Chat-t
   * `model.py`: F2Chat-t model
   * `config.json`: a json file including settings for F2Chat-t
   * `reader_stage2.py`: dataset reader for F2Chat
   * `model_stage2.py`: F2Chat model
   * `config_stage2.json`: a json file including settings for F2Chat
   * `siamese_metric.py`: validation metric for training siamese network
   * `util.py`: util fuctions, e.g., replacement of special tokens, generation of train set and test set.
   * `data/`: all the datasets (in json format)

      * `angular.json`: 989 threads from Angular room on Gitter.
      * `spring-boot.json`: 985 threads from Spring-boot room on Gitter.
      * `deeplearning4j.json`: 985 threads from Deeplearning4j room on Gitter.

      *We use fold 0 of deeplearning4j as an example. All the datasets used in the experiments can be generated using functions in [util.py](util.py). `<dataset>_1.json` is the corresponding dataset with handcrafted non-textual features.*

* `F2Chat_s/`: code of F2Chat-s. This is for our ablation study. F2Chat-s **S**imultaneously train both textual and non-textual encoders using siamese network.
   * `reader.py`: dataset reader.
   * `model.py`: F2Chat-s model
   * `siamese_metric.py`: validation metric for training siamese network
   * `util.py`: util fuctions, e.g., replacement of special tokens, generation of train set and test set.
   * `config.json`: a json file including settings.

### Train & Test

Open terminal in the parent folder and run
``allennlp train <config file> -s <serialization path> -f --include-package <package name>``. Please refer to official docs of [AllenNLP](https://allennlp.org/) for more details.

For example, with `allennlp train F2Chat/config.json -s F2Chat/out/ -f --include-package F2Chat`, you can get the output folder at `F2Chat/out/` and log information showed on the console. There are three packages (i.e., `Baseline`, `F2Chat` and `F2Chat_s`) available, with each one corresponds to a model.

For test, please follow the comments in [predict.py](predict.py) and [predict_stage2.py](predict_stage2.py). You will get Precision, Recall and F1-score for each categorey and average metrics weighted by Support. All the metrics `<project>_metrics_<fold>.json` and detailed results of each sample `<project>_results_<fold>.json` are in `./<package name>/data/`.
