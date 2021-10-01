import json
import random
import re
import numpy as np
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional
import logging
from datetime import datetime
from copy import deepcopy

from allennlp.data import Field
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from .util import Processer

logger = logging.getLogger(__name__)

@DatasetReader.register("reader_stage2")
class ReaderStage2(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_PTM: bool = False,
                 max_tokens: int = 50,
                 max_messages: int = 20,
                 use_numeric: bool = True,
                 cache_directory: Optional[str] = None,
                 merge: bool = False,
                 text_only: bool = True,
                 use_level1: bool = False,
                 processed: bool = False,
                 stat_path: str = None) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory)

        self._token_indexers = token_indexers  # token indexers for text
        self._processer = Processer(tokenizer=tokenizer,
                                    max_messages=max_messages,
                                    max_tokens=max_tokens,
                                    merge=merge,
                                    text_only=text_only,
                                    use_numeric=use_numeric,
                                    use_PTM=use_PTM,
                                    processed=processed)
        self._text_only = text_only
        self._use_level1 = use_level1
        self._mapping = {'programming_problem': 'problem_report', 'library_problem': 'problem_report', 'documentation_problem': 'problem_report',
            'programming_information': 'information_retrieval', 'library_information': 'information_retrieval', 'documentation_information': 'information_retrieval', 'general_information': 'information_retrieval',
            'technical_discussion': 'project_management', 'task_progress': 'project_management'}
        if not text_only:
            # all non-textual features are prepared before reading
            self._use_numeric = use_numeric
            if use_numeric:
                # only used to determine whether to use a feature or not
                self._stat = json.load(open(stat_path, 'r', encoding="utf-8"))
        
    def read_dataset(self, file_path):
        dialogues = json.load(open(file_path, 'r', encoding="utf-8"))
        dataset = dict()
        for dia in dialogues:
            if self._use_level1:
                dia["intention"] = self._mapping[dia["intention"]]
            label = dia["intention"]
            if dataset.get(label) is None:
                dataset[label] = list()
            dataset[label].append(self._processer.process(dia))
            # dataset[label].append(dia)
        if not self._text_only and self._use_numeric:
            pass
            '''
            # already normalized
            for _ in dataset.keys():
                for dia in dataset[_]:
                    for fea, v in dia["dia_fea"].items():
                        if self._stat[fea]["use"]:
                            # cut
                            if v <= self._stat[fea]["min"]:
                                dia["dia_fea"][fea] = 0
                            elif v >= self._stat[fea]["max"]:
                                dia["dia_fea"][fea] = 1
                            else:
                                dia["dia_fea"][fea] = (v - self._stat[fea]["min"]) / (self._stat[fea]["max"] - self._stat[fea]["min"])
            '''
        return dataset

    @overrides
    def _read(self, file_path):
        dataset = self.read_dataset(file_path)
        all_data = []
        for ll in list(dataset.values()):
            all_data.extend(ll)

        intentions = list(dataset.keys())
        logger.info(intentions)

        if "test" in file_path:
            logger.info("loading unlabel examples ...")
            for sample in all_data:
                yield self.text_to_instance(sample, type_="unlabel")
            logger.info(f"Num of unlabel instances is {len(all_data)}")

        elif "validation" in file_path:
            logger.info("loading testing examples ...")
            for sample in all_data:
                yield self.text_to_instance(sample, type_="test")
            logger.info(f"Num of testing instances is {len(all_data)}")
            
        else:
            random.shuffle(all_data)
            # training
            logger.info("loading training examples ...")
            for sample in all_data:
                yield self.text_to_instance(sample, type_="train")
            logger.info(f"Num of training instances is {len(all_data)}")

    @overrides
    def text_to_instance(self, ins, type_="train") -> Instance:  # type: ignore
        # share the code between predictor and trainer, hence the label field is optional
        fields: Dict[str, Field] = {}

        fields["dialog"] = ListField([TextField(line, self._token_indexers) for line in ins["mess_text"]])

        if not self._text_only:
            # dialogue level feature
            if self._use_numeric:
                fields["nontext_feature_dia"] = ArrayField(np.array([ins["dia_fea"][fea] for fea in ins["dia_fea"].keys() if self._stat[fea]["use"]]))
            else:
                # not implemented
                fields["nontext_feature_dia"] = ListField([LabelField(ins["dia_fea"][fea], label_namespace=f"{fea}_dia_features") for fea in ins["dia_fea"].keys() if self._stat[fea]["use"]])

            '''
            # message level feature
            # not implemented
            for feature in self._message_features:
                fields[f"{feature}_mess"] = LabelField(ins["mess_fea"][feature], label_namespace=f"{feature}_mess_features")
            '''

        fields['label'] = LabelField(ins["intention"], label_namespace="intention_labels")
        
        meta_ins = {"id": ins["id"], "project": ins["project"]}
        fields['metadata'] = MetadataField({"type": type_, "instance": meta_ins})

        return Instance(fields)