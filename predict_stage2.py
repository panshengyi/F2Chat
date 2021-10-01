import torch

from torch.utils.data import DataLoader
import re
from allennlp.data import Instance, Token, Vocabulary, allennlp_collate
from allennlp.data.token_indexers import SingleIdTokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedTransformerEmbedder

from allennlp.common import Params
from allennlp.models import Model
import json
import importlib
import numpy as np
from typing import Dict, Any
import random

def cal_metrics(predict_result):
    sample_num = len(predict_result)
    intentions = set([_["intention"] for _ in predict_result.values()])
    metrics = {}
    for label in intentions:
        metrics[label] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "support": 0}
    
    correct = []
    false = []

    for dialogue in predict_result.values():
        ground_truth = dialogue["intention"]
        predict = dialogue["predict_label"]
        metrics[ground_truth]["support"] += 1
        # print(ground_truth, predict)
        if ground_truth == predict:
            correct.append(dialogue)
            for label in metrics.keys():
                if label == ground_truth:
                    metrics[label]["TP"] += 1
                else:
                    metrics[label]["TN"] += 1
        else:
            false.append(dialogue)
            for label in metrics.keys():
                if label == ground_truth:
                    metrics[label]["FN"] += 1
                elif label == predict:
                    metrics[label]["FP"] += 1
                else:
                    metrics[label]["TN"] += 1
    
    eps = 1e-10
    for label in intentions:
        metrics[label]["precision"] = metrics[label]["TP"] / (metrics[label]["TP"] + metrics[label]["FP"] + eps)
        metrics[label]["recall"] = metrics[label]["TP"] / (metrics[label]["TP"] + metrics[label]["FN"] + eps)
        metrics[label]["F1-score"] = 2 * metrics[label]["TP"] / (2 * metrics[label]["TP"] + metrics[label]["FP"] + metrics[label]["FN"] + eps)
        metrics[label]["accuracy"] = (metrics[label]["TP"] + metrics[label]["TN"]) / (metrics[label]["TP"] + metrics[label]["TN"] + metrics[label]["FP"] + metrics[label]["FN"])
    
    # add overall metrics
    metrics["accuracy"] = len(correct) / (len(correct) + len(false))
    for m in ["precision", "recall", "F1-score"]:
        metrics[m] = 0
        for label in intentions:
            metrics[m] += metrics[label][m] * metrics[label]["support"]
        metrics[m] /= sample_num
    
    return correct, false, metrics


def test(room, fold, serialization, model="F2Chat", weights="best.th", merge=True, level1=False, max_tokens=128, max_messages=20, seed=2021, use_PTM=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    path = f"./{model}"  # path for model
    datapath = "./F2Chat"

    importlib.import_module(".model_stage2", package=model)  # model for stage 2
    importlib.import_module(".model", package=model)  # model for stage 1
    reader = importlib.import_module(".reader_stage2", package=model)

    predict_path = datapath + f"/data/{room}_test_{fold}_1.json"
    
    if use_PTM:
        token_indexers = {"tokens": PretrainedTransformerIndexer(PTM, namespace="tags")}
        tokenizer = PretrainedTransformerTokenizer(PTM, add_special_tokens=False, max_length=50)
    else:
        token_indexers = {"tokens": SingleIdTokenIndexer(namespace="token_vocab", lowercase_tokens=True),
                          "pos_tags":SingleIdTokenIndexer(namespace="pos_tag_vocab", feature_name="tag_")}
        tokenizer = SpacyTokenizer(pos_tags=True)

    stat_path = datapath + f"/data/{room}_minmax.json"
    dataset_reader = reader.ReaderStage2(tokenizer=tokenizer, token_indexers=token_indexers, max_tokens=max_tokens, max_messages=max_messages, merge=merge, text_only=False, use_numeric=True, use_PTM=True, stat_path=stat_path, use_level1=level1, processed=True)
    predict_dataset = dataset_reader.read(predict_path).instances  # list

    # load the trained model
    serialization_path = path + f"/{serialization}"
    config_path = serialization_path + "/config.json"
    params = Params(json.load(open(config_path, "r")))
    model = Model.load(params, serialization_path, weights_file=serialization_path + f"/{weights}", cuda_device=-1)
    model.eval()
    # vocab = model.vocab

    output_dict = model.forward_on_instances(predict_dataset[:110])  # get prediction result
    # large dataset can not be tested at the same time due to the limitation of memory
    len_predict_dataset = len(predict_dataset)
    for i in range(int(len_predict_dataset / 110)):
        output_dict.extend(model.forward_on_instances(predict_dataset[110*(i+1):110*(i+2)]))
    
    # metric = model.get_metrics()
    # with open(path + f"/{room}_metrics.json", "w", encoding="UTF-8") as f:
        # json.dump(metric, f, indent=4)

    map_ = MAP_1 if level1 else MAP_2

    predict_dialogues = json.load(open(predict_path, "r", encoding="UTF-8"))
    result = {}
    for dia in predict_dialogues:
        dia["intention"] = map_[dia["intention"]]
        result[f"{dia['project']}_{dia['id']}"] = dia

    for idx, instance in enumerate(output_dict):
        prediction = instance["label"]
        dia = instance["meta"]["instance"]
        # print(result[dia[0]["id"]]["intention"], vote_result[0][0])
        # print(vote_result)
        result[f"{dia['project']}_{dia['id']}"]["predict_label"] = prediction
        # result[dia[0]["id"]]["vote_record"] = vote_result
    
    correct, false, metrics = cal_metrics(result)
    print(f"correct: {len(correct)}, false: {len(false)}")
    print(metrics["accuracy"])

    with open(path + f"/data/{room}_metrics_{fold}_stage2.json", "w", encoding="UTF-8") as f:
        json.dump(metrics, f, indent=4)
    
    with open(path + f"/data/{room}_results_{fold}_stage2.json", "w", encoding="UTF-8") as f:
        json.dump(result, f, indent=4)
    
    print("finished predicting ...")


def cal_average_results(room):
    serialization_path = "xxx"
    test_metric = dict()
    metric_name = ["accuracy", "precision", "recall", "F1-score"]
    information_types = list(MAP_2.keys())
    for i in range(0, 5):
        with open(serialization_path + f"{room}_metrics_{i}_stage2.json", 'r') as f:
            metrics = json.load(f)
            test_metric[i] = metrics
    
    average_metric = dict()
    for info in information_types:
        average_metric[info] = dict()
        for m in metric_name:
            tmp = list()
            for i in range(0, 5):
                tmp.append(test_metric[i][info][m])
            average_metric[info][m] = np.average(tmp)
    
    for m in metric_name:
        tmp = list()
        for i in range(0, 5):
            tmp.append(test_metric[i][m])
        average_metric[m] = np.average(tmp)

    print(average_metric)
    

# level 1
MAP_1 = {'programming_problem': 'problem_report', 'library_problem': 'problem_report', 'documentation_problem': 'problem_report',
            'programming_information': 'information_retrieval', 'library_information': 'information_retrieval', 'documentation_information': 'information_retrieval', 'general_information': 'information_retrieval',
            'technical_discussion': 'project_management', 'task_progress': 'project_management'}

# level 2
MAP_2= {'programming_problem': 'programming_problem', 'library_problem': 'library_problem', 'documentation_problem': 'documentation_problem',
            'programming_information': 'programming_information', 'library_information': 'library_information', 'documentation_information': 'documentation_information', 'general_information': 'general_information',
            'technical_discussion': 'technical_discussion', 'task_progress': 'task_progress'}

PTM = "google/bert_uncased_L-4_H-512_A-8"

if __name__ == "__main__":
    room = "deeplearning4j"
    fold = 0
    serialization = "out_deeplearning4j_0_stage2"
    model = "F2Chat"
    weights = "best.th"
    # max_tokens is used to limit the length of merged messages. Message length before merging is limited by the PTM tokenizer (set to 50).
    test(room, fold, serialization, model, weights, max_tokens=128, max_messages=20, merge=True, level1=False, seed=2021, use_PTM=True)
