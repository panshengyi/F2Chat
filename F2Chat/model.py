import logging
from typing import Dict, List, Any

# from allennlp.modules.similarity_functions import BilinearSimilarity
from overrides import overrides

import torch
import numpy as np
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder
from allennlp.modules.seq2vec_encoders import CnnEncoder, BagOfEmbeddingsEncoder, BertPooler
from allennlp.nn import RegularizerApplicator, InitializerApplicator, Activation
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, F1Measure, Metric
from allennlp.training.util import get_batch_size

from torch import nn
from torch.nn import Dropout, PairwiseDistance, CosineSimilarity
import torch.nn.functional as F
from torch.autograd import Variable

from .siamese_metric import SiameseMeasureV1, SiameseMeasureV2
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
        # print(alpha.size())
        # print(probs.size())
        # log_p = probs.log()
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * probs

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # print(loss.shape)
        return loss


@Model.register("f2chat_t")
class F2ChatT(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 PTM: str = 'bert-base-uncased',
                 additional_token_num: int = 0,
                 dropout: float = 0.1,
                 label_namespace: str = "labels",
                 class_weights: str = None,
                 device: str = "cpu",
                 use_header: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None) -> None:
        
        super().__init__(vocab, regularizer)
        
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)
        self._use_header = use_header

        self._label_namespace = label_namespace
        self._idx2token_label = vocab.get_index_to_token_vocabulary(namespace=label_namespace)
        self._dropout = Dropout(dropout)

        '''
        # we use extra token to replace the [UNK] in BERT's vocab, thus there is no need to add here 
        if additional_token_num != 0:
            bert_token_num = 30522
            embedding = getattr(text_field_embedder, f"token_embedder_{list(text_field_embedder._token_embedders.keys())[0]}")
            embedding.transformer_model.resize_token_embeddings(bert_token_num + additional_token_num)
        '''

        self._text_field_embedder = text_field_embedder
        self._bert_pooler = BertPooler(PTM, requires_grad=True, dropout=dropout)
        
        lstm_input_dim = self._bert_pooler.get_output_dim()
        
        '''
        rnn = nn.LSTM(input_size=lstm_input_dim,
                      hidden_size=int(lstm_input_dim / 2),
                      num_layers=1,
                      dropout=0,
                      batch_first=True,
                      bidirectional=True)
        '''
        self._encoder = LstmSeq2SeqEncoder(input_size=lstm_input_dim,
                                           hidden_size=int(lstm_input_dim / 2),
                                           num_layers=1,
                                           dropout=0.0,
                                           bidirectional=True
                                           )
        # self._encoder = MultiHeadSelfAttention(4, 100, 100, 100)
        embedding_dim = self._encoder.get_output_dim()  # use maxpooling as seq2vec module

        self._num_class = self.vocab.get_vocab_size(self._label_namespace)
        self._num_class_tag = self.vocab.get_vocab_size("label_tags")
        # self._same_class_idx = self.vocab.get_token_index("same", "labels")
        # self._bilinear_sim = BilinearSimilarity(self._encoder.get_output_dim(), self._encoder.get_output_dim())
        if use_header:
            self._projector_single = ProjectionHeader(embedding_dim, int(embedding_dim / 2), 1, [embedding_dim], torch.nn.ReLU(), dropout)
            embedding_dim = int(embedding_dim / 2)
        # similarity measure
        self._projector = ProjectionHeader(embedding_dim*2, self._num_class, 1, [embedding_dim], torch.nn.ReLU(), dropout)

        self._golden_instances_embeddings = None  # embeddings for golden instances
        self._golden_instances_labels = None  # labels for golden instances
        self._golden_instances_ids = None  # dialogue id

        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1-score_overall": FBetaMeasure(beta=1.0, average="weighted", labels=range(self._num_class)),  # return float
            "f1-score_each": FBetaMeasure(beta=1.0, average=None, labels=range(self._num_class))  # return list[float]
        }
        self._siamese_metirc_1 = SiameseMeasureV1(self.vocab)  # V1 or V2
        self._siamese_metirc_2 = SiameseMeasureV2(self.vocab)

        '''
        self._class_weights = torch.ones(self._num_class_tag, 1).to(self.device)
        if class_weights is not None:
            class_weights = list(json.load(open(class_weights, "r", encoding="UTF-8")).values())
            assert self._num_class_tag == len(class_weights)
            self._class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device).view(-1, 1)
        '''

        # self._loss = MultiCEFocalLoss(label_num=self._num_class, label_tag_num=self._num_class_tag, gamma=2, alpha=self._class_weights)
        self._loss = torch.nn.CrossEntropyLoss()
        # self._mse_loss = torch.nn.MSELoss()
        initializer(self)

    @staticmethod
    def contrastive_loss(left, right):
        pairwise_distance = torch.nn.PairwiseDistance(p=1).cuda()
        return torch.exp(-pairwise_distance(left, right)).cuda()

    def _instance_forward(self,
                          dialog: TextFieldTensors,
                          use_header: bool = False):

        assert not use_header or hasattr(self, "_projector_single")
        # print(dialog["tokens"]["token_ids"].shape)
        dialog_embedder = self._text_field_embedder(dialog, num_wrapping_dims=1)  # listfield must specify num_wrapping_dims
        # print(dialog_embedder.shape)
        dialog_shape = dialog_embedder.shape
        dialog_embedder = dialog_embedder.view(-1, dialog_shape[-2], dialog_shape[-1])
        # dialog_embedder = self._dropout(dialog_embedder)
        dialog_out = self._bert_pooler(dialog_embedder)  # batch_size*mess_num*embedding_dim
        dialog_out = dialog_out.view(dialog_shape[0], dialog_shape[1], -1)
        
        dialog_mask = get_text_field_mask(dialog, num_wrapping_dims=1).float()

        dialog_mask = torch.sum(dialog_mask, -1) > 0  # batch_size*num_messages only if all the tokens in the message is padded, then this message is padded
        rnn_out = self._encoder(dialog_out, dialog_mask)  # batch_size*num_messages*300
        rnn_out = rnn_out.transpose(1, 2)
        rnn2vec = F.max_pool1d(rnn_out, kernel_size=rnn_out.shape[-1])
        rnn2vec = rnn2vec.squeeze(-1)
        if use_header:
            rnn2vec = self._projector_single(rnn2vec)
        return rnn2vec

    def forward_gold_instances(self, d_id, dialog, label_tags):
        embedding = self._instance_forward(dialog, use_header=self._use_header)
        if not torch.is_tensor(self._golden_instances_embeddings):
            self._golden_instances_embeddings = embedding
            self._golden_instances_ids = d_id
            self._golden_instances_labels = [self.vocab.get_token_from_index(_, namespace="label_tags").split('@')[0] for _ in label_tags.tolist()]
                
        else:
            self._golden_instances_embeddings = torch.cat([self._golden_instances_embeddings, embedding])
            self._golden_instances_ids.extend(d_id)
            self._golden_instances_labels.extend([self.vocab.get_token_from_index(_, namespace="label_tags").split('@')[0] for _ in label_tags.tolist()])

    def forward(self,
                dialog1: TextFieldTensors = None,
                dialog2: TextFieldTensors = None,
                label: torch.IntTensor = None,
                label_tags: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        # print(dialog1["tokens"]["tokens"].shape)
        output_dict = dict()
        if metadata:
            output_dict["meta"] = metadata
        
        if metadata and metadata[0]["type"] == "golden":
            self.forward_gold_instances([ _["instance"][0]['id'] for _ in metadata], dialog1, label_tags)
            return output_dict

        rnn_vec1 = self._instance_forward(dialog1, use_header=self._use_header)
        if metadata and metadata[0]["type"] == "unlabel":
            # unlabel
            logits = []
            for instance in rnn_vec1:
                logits.append(self._projector(torch.cat([instance.expand(len(self._golden_instances_ids), -1), self._golden_instances_embeddings], -1)))
            output_dict['logits'] = logits
            output_dict['probs'] = [nn.functional.softmax(_, dim=-1) for _ in logits]
            return output_dict

        # type == "train" or type == "test"
        rnn_vec2 = self._instance_forward(dialog2, use_header=self._use_header)
        logits = self._projector(torch.cat([rnn_vec1, rnn_vec2], -1))
        probs = nn.functional.softmax(logits, dim=-1)
        output_dict["logits"] = logits
        output_dict["probs"] = probs
        loss = self._loss(logits, label)
        output_dict['loss'] = loss
        output_dict['label_tags'] = label_tags
        for metric_name, metric in self._metrics.items():
            metric(predictions=probs, gold_labels=label)
        self._siamese_metirc_1(probs, label_tags)  # metric defined by ourselves
        self._siamese_metirc_2(probs, label_tags)
        return output_dict

    def make_output_human_readable(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "meta" not in output_dict or output_dict["meta"][0]["type"] != "unlabel":
            return output_dict
        
        '''
        # there are three voting strategies, choose any one
        # vote based on the decision
        classes = []
        for probs in output_dict['probs']:
            c = []
            for p in probs:
                label_idx = p.argmax(dim=-1).item()
                label_str = self.vocab.get_token_from_index(label_idx, namespace="labels")
                c.append(label_str)
            classes.append(c)
        
        vote_num = {}
        output_dict['label'] = []
        for intention in set(self._golden_instances_labels):
            vote_num[intention] = 0
        
        for c in classes:
            # reset vote_num
            for key in vote_num.keys():
                vote_num[key] = 0
            for class_name, golden_name in zip(c, self._golden_instances_labels):
                if class_name == "same":
                    vote_num[golden_name] += 1
                else:
                    vote_num[golden_name] -= 0.3
            # print(vote_num)
            output_dict['label'].append(deepcopy(vote_num))
        '''

        '''
        # vote based on the mean score
        vote_num = {}
        output_dict['label'] = []
        for intention in set(self._golden_instances_labels):
            vote_num[intention] = 0

        idx_same = self.vocab.get_token_index("same", namespace="labels")
        for probs in output_dict['probs']:
            for key in vote_num.keys():
                vote_num[key] = 0
            for p, golden_name in zip(probs, self._golden_instances_labels):
                vote_num[golden_name] += p[idx_same].item()
            output_dict['label'].append(deepcopy(vote_num))
        '''

        # vote based on the max score
        vote_num = {}
        output_dict['label'] = []
        for intention in set(self._golden_instances_labels):
            vote_num[intention] = 0

        idx_same = self.vocab.get_token_index("same", namespace=self._label_namespace)
        for probs in output_dict['probs']:
            for key in vote_num.keys():
                vote_num[key] = 0
            for p, golden_name in zip(probs, self._golden_instances_labels):
                if p[idx_same].item() > vote_num[golden_name]:
                    vote_num[golden_name] = p[idx_same].item()
            output_dict['label'].append(deepcopy(vote_num))

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
        metrics["siamese_score_1"] = self._siamese_metirc_1.get_metric(reset)
        metrics["siamese_score_2"] = self._siamese_metirc_2.get_metric(reset)
        return metrics
    
    def get_output_dim(self, use_header=False):
        assert not use_header or hasattr(self, "_projector_single")
        if use_header:
            return self._projector_single.get_output_dim()
        return self._seq2vec.get_output_dim()