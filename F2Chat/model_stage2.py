import logging
from typing import Dict, List, Any, Union

# from allennlp.modules.similarity_functions import BilinearSimilarity
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.common import Params
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric
from allennlp.training.util import get_batch_size

from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable

from .util import pack2sequence, ProjectionHeader

import warnings
import json
from copy import deepcopy

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, label_num, label_tag_num, gamma=2, alpha=None, temperature=1, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(label_tag_num, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.temperature = temperature
        self.label_num = label_num
        self.label_tag_num = label_tag_num

    def forward(self, predict, labels, label_tags):
        # Focal loss: -alpha*(1-prob)^gamma*log(prob)
        # pt = F.softmax(predict / self.temperature, dim=-1)
        pt = F.log_softmax(predict / self.temperature, dim=-1)  # softmax+log
        class_mask = F.one_hot(labels, self.label_num)  # one-hot label
        ids = label_tags.detach().view(-1)
        alpha = self.alpha[ids] 
        probs = (pt * class_mask).sum(1).view(-1, 1)
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * probs

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

@Model.register("f2chat")
class F2Chat(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 encoder_path: str,
                 encoder_weights: str = "best.th",
                 use_encoder_header: bool = False,
                 use_numeric: bool = False,
                 nontext_header: int = None,
                 text_header: int = None,
                 text_only: bool = True,
                 stat_path: str = None,
                 dropout: float = 0.1,
                 label_namespace: str = "intention_labels",
                 class_weights: str = None,
                 device: str = "cpu",   
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        super().__init__(vocab, regularizer)

        assert text_only or stat_path
        # self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self._device = torch.device(device)
        self._label_namespace = label_namespace
        self._dropout = Dropout(dropout)
        self._text_only = text_only
        self._nontext_header = nontext_header
        self._text_header = text_header
        self._use_encoder_header = use_encoder_header
        
        self._idx2token_label = vocab.get_index_to_token_vocabulary(namespace=label_namespace)
        self._encoder = Model.load(Params.from_file(encoder_path + "config.json"), encoder_path, 
                                    weights_file = encoder_path + encoder_weights, 
                                    cuda_device = int(device.split(":")[-1]) if device != 'cpu' else -1)  # vocabulary will also be loaded
        # vocab = model.vocab
        # self._encoder.vocab.extend_from_vocab(vocab)
        # self._encoder.extend_embedder_vocab()
        text_embedding_dim = self._encoder.get_output_dim(use_header=use_encoder_header)
        if text_header:
            # whether to use an extra header for textual feature
            self._projector_text = ProjectionHeader(text_embedding_dim, text_header, 1, [text_header], torch.nn.ReLU(), dropout)
            text_embedding_dim = text_header

        nontext_embedding_dim = 0
        if not text_only:
            self._use_numeric = use_numeric
            self._stat = json.load(open(stat_path, 'r', encoding="utf-8"))
            if use_numeric:
                nontext_embedding_dim = len([_ for _ in self._stat.keys() if self._stat[_]["use"]])
            else:
                for fea, dim in self._stat.items():
                    if self._stat[fea]["use"]:
                        setattr(self, f"{fea}_embedder", Embedding(embedding_dim=self._stat[fea]["embedding_dim"], vocab_namespace=f"{fea}_dia_features", vocab=vocab).to(self._device))
                    nontext_embedding_dim += self._stat[fea]["embedding_dim"]
        
            if nontext_header:
                self._projector_nontext = ProjectionHeader(nontext_embedding_dim, nontext_header, 1, [nontext_header], torch.nn.ReLU(), dropout)
                nontext_embedding_dim = nontext_header
        
        hidden_dim = 128
        self._num_class = self.vocab.get_vocab_size(self._label_namespace)
        self._projector = ProjectionHeader(text_embedding_dim + nontext_embedding_dim, self._num_class, 1, [hidden_dim], torch.nn.ReLU(), dropout)

        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1-score_overall": FBetaMeasure(beta=1.0, average="weighted", labels=range(self._num_class)),  # return float
            "f1-score_each": FBetaMeasure(beta=1.0, average=None, labels=range(self._num_class))  # return list[float]
        }
        # self._siamese_metirc = SiameseMeasure(self.vocab)
        loss_weights = {'programming_problem': 1, 'library_problem': 1, 'documentation_problem': 1,
                         'programming_information': 1, 'library_information': 1, 'documentation_information': 1, 'general_information': 1,
                         'technical_discussion': 1, 'task_progress': 1}
        if class_weights:
            loss_weights = json.load(open(class_weights, "r", encoding="UTF-8"))
            assert self._num_class == len(loss_weights)

        self._class_weights = list()
        for idx in range(self._num_class):
            self._class_weights.append(loss_weights[self._idx2token_label[idx]])
        self._class_weights = torch.tensor(self._class_weights, dtype=torch.float32).to(self._device)

        # self._loss = MultiCEFocalLoss(label_num=self._num_class, label_tag_num=self._num_class_tag, gamma=2, alpha=self._class_weights)
        self._loss = torch.nn.CrossEntropyLoss(weight=self._class_weights)
        # self._contrastive_loss = ContrastiveLoss()
        # self._mse_loss = torch.nn.MSELoss()
        initializer(self)

    def forward(self,
                dialog: TextFieldTensors,
                nontext_feature_dia: torch.Tensor = None,
                label: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:

        output_dict = dict()
        if metadata:
            output_dict["meta"] = metadata
        text_embedding = self._encoder._instance_forward(dialog, use_header=self._use_encoder_header)  # get textual feature
        if self._text_header:
            text_embedding = self._projector_text(text_embedding)
        embedding = text_embedding

        nontext_embedding = None
        if not self._text_only:
            if self._use_numeric:
                nontext_embedding = nontext_feature_dia
            else:
                idx = 0
                for fea in self._stat.keys():
                    if self._stat[fea]["use"]:
                        tensor = nontext_feature_dia[idx]
                        embedder = getattr(self, f"{fea}_embedder")
                        if torch.is_tensor(nontext_embedding):
                            nontext_embedding = torch.cat([nontext_embedding, embedder(tensor)], -1)
                        else:
                            nontext_embedding = embedder(tensor)
            
            if self._nontext_header:
                nontext_embedding = self._projector_nontext(nontext_embedding)
            
            embedding = torch.cat([text_embedding, nontext_embedding], -1)  # concat

        logits = self._projector(embedding)

        probs = nn.functional.softmax(logits, dim=-1)
        output_dict["logits"] = logits
        output_dict["probs"] = probs
        if metadata[0]["type"] != "unlabel":
            loss = self._loss(logits, label)
            output_dict['loss'] = loss
            for metric_name, metric in self._metrics.items():
                metric(predictions=logits, gold_labels=label)

        return output_dict
    
    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        idx = torch.argmax(output_dict["probs"], dim=1).tolist()
        output_dict["label"] = [self._idx2token_label[_] for _ in idx]
        '''
        for _, label in enumerate(output_dict["label"]):
            if label == "team_discussion":
                print(output_dict["probs"][_])
                print(output_dict["probs"][_][4], output_dict["probs"][_][1])
                print("=============================")
        '''
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = dict()
        metrics['accuracy'] = self._metrics['accuracy'].get_metric(reset)
        precision, recall, fscore = self._metrics['f1-score_overall'].get_metric(reset).values()
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1-score'] = fscore
        precision, recall, fscore = self._metrics['f1-score_each'].get_metric(reset).values()
        for i in range(self._num_class):
            metrics[f'{self._idx2token_label[i]}_precision'] = precision[i]
            metrics[f'{self._idx2token_label[i]}_recall'] = recall[i]
            metrics[f'{self._idx2token_label[i]}_f1-score'] = fscore[i]

        return metrics