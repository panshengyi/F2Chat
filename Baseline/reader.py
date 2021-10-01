import json
import random
import re
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
from allennlp.data.fields import LabelField, TextField, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from .util import Processer

logger = logging.getLogger(__name__)

@DatasetReader.register("reader")
class Reader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 use_PTM: bool = False,
                 max_tokens: int = 50,
                 max_messages: int = 20,
                 same_diff_ratio: Dict[str, int] = None,
                 cache_directory: Optional[str] = None,
                 merge: bool = False,
                 text_only: bool = True,
                 use_level1: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=lazy, cache_directory=cache_directory)
        # self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._processer = Processer(tokenizer, max_messages, max_tokens, merge, text_only)
        self._same_diff_ratio = same_diff_ratio or {"diff": {"intention": 4, "num_per_intention": 1}, "same": 4}  # positive:negative=1:1 as suggested in FRMiner
        self._text_only = text_only
        self._use_level1 = use_level1
        self._mapping = {'programming_problem': 'problem_report', 'library_problem': 'problem_report', 'documentation_problem': 'problem_report',
            'programming_information': 'information_retrieval', 'library_information': 'information_retrieval', 'documentation_information': 'information_retrieval', 'general_information': 'information_retrieval',
            'technical_discussion': 'project_management', 'task_progress': 'project_management'}

    def read_dataset(self, file_path):
        dialogues = json.load(open(file_path, 'r', encoding="utf-8"))

        dataset = {}
        for dia in dialogues:
            if self._use_level1:
                dia["intention"] = self._mapping[dia["intention"]]     
            label = dia["intention"]
            if dataset.get(label) is None:
                dataset[label] = []

            dataset[label].append(self._processer.process(dia))
            # dataset[label].append(dia)

        return dataset

    @overrides
    def _read(self, file_path):
        dataset = self.read_dataset(file_path)
        all_data = []
        min_sample_num = 9999
        for ll in list(dataset.values()):
            all_data.extend(ll)
            if len(ll) < min_sample_num:
                min_sample_num = len(ll)
        random.shuffle(all_data)

        intentions = list(dataset.keys())
        intention_num = len(intentions)
        logger.info(intentions)
        same_num = 0
        diff_num = 0

        if "golden" in file_path:
            logger.info("Begin loading golden instances------")
            for sample in all_data:
                yield self.text_to_instance((sample, sample), type_="golden")
            logger.info(f"Num of golden instances is {len(all_data)}")

        elif "test" in file_path:
            logger.info("Begin predict------")
            for sample in all_data:
                yield self.text_to_instance((sample, sample), type_="unlabel")
            logger.info(f"Predict sample num is {len(all_data)}")

        elif "validation" in file_path:
            logger.info("Begin testing------")
            dataset = self.read_dataset(re.sub("validation", "train", file_path))
            iter_num = 3
            for _ in range(iter_num):
                for sample in all_data:
                    key = sample["intention"]
                    yield self.text_to_instance((sample, random.choice(dataset[key])), type_="test")
                    for intention, value in dataset.items():
                        if intention == key:
                            continue
                        yield self.text_to_instance((sample, random.choice(value)), type_="test")
                    
                    same_num += 1
                    diff_num += (intention_num - 1)
            
            logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")

        else:
            iter_num = 5

            same_per_sample = self._same_diff_ratio["same"]
            diff_intention = self._same_diff_ratio["diff"]["intention"]
            diff_per_intention = self._same_diff_ratio["diff"]["num_per_intention"]
            diff_per_sample = diff_intention * diff_per_intention

            assert diff_intention + 1 <= intention_num
            assert max(same_per_sample, diff_per_intention) <= min_sample_num

            for _ in range(iter_num):
                for sample in all_data:
                    tmp_diff = diff_intention
                    key = sample["intention"]
                    selected_intention = random.sample(intentions, k = diff_intention + 1)
                    for same in random.sample(dataset[key], k=same_per_sample):
                        yield self.text_to_instance((sample, same))
                    for intention in selected_intention:
                        if intention == key:
                            continue
                        else:
                            for diff in random.sample(dataset[intention], k = diff_per_intention):
                                yield self.text_to_instance((sample, diff), type_="train")
                            tmp_diff -= 1
                            if tmp_diff == 0:
                                break

                    same_num += same_per_sample
                    diff_num += diff_per_sample
            
            logger.info(f"Dataset Count: Same : {same_num} / Diff : {diff_num}")

    @overrides
    def text_to_instance(self, p, type_="train") -> Instance:
        # type_ has 4 options: train, test, unlabel, golden
        fields: Dict[str, Field] = {}
        ins1, ins2 = p # instance:Dict{id, intention, messages} mess:Dict{id, text, time, index, user}

        fields["dialog1"] = ListField([TextField(line, self._token_indexers) for line in ins1["mess_text"]])
        if type_ in ["train", "test"]:
            fields["dialog2"] = ListField([TextField(line, self._token_indexers) for line in ins2["mess_text"]])
        
        if not self._text_only:
            pass
        
        fields['label_tags'] = LabelField("@".join([ins1["intention"], ins2["intention"]]), label_namespace="label_tags")
        if type_ in ["train", "test"]:
            if ins1["intention"] == ins2["intention"]:
                fields['label'] = LabelField("same")
            else:
                fields['label'] = LabelField("diff")
            
        fields['metadata'] = MetadataField({"type": type_, "instance": p})

        return Instance(fields)