import torch
import json
import random
import numpy as np
from math import log10, log
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, get_mask_from_sequence_lengths, sort_batch_by_length
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import WhitespaceTokenizer, SpacyTokenizer
import re
from datetime import datetime
import matplotlib.pyplot as plt
import copy

from typing import List, Union
from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation


class ProjectionHeader(torch.nn.Module, FromParams):
    # reconstruct FeedForward Module in Allennlp (the output layer is without activation and bias)
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_dims: Union[int, List[int]],
        activations: Union[Activation, List[Activation]],
        dropout: Union[float, List[float]] = 0.0,
        use_bias: bool = False,
    ) -> None:

        super().__init__()
        # num_layers: the numer of hidden layers (not include the input and output layers)

        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers
        if not isinstance(activations, list):
            activations = [activations] * num_layers
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers
        if len(hidden_dims) != num_layers:
            raise ConfigurationError(
                "len(hidden_dims) (%d) != num_layers (%d)" % (len(hidden_dims), num_layers)
            )
        if len(activations) != num_layers:
            raise ConfigurationError(
                "len(activations) (%d) != num_layers (%d)" % (len(activations), num_layers)
            )
        if len(dropout) != num_layers:
            raise ConfigurationError(
                "len(dropout) (%d) != num_layers (%d)" % (len(dropout), num_layers)
            )
        self._activations = torch.nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = output_dim
        self.input_dim = input_dim
        if num_layers == 0:
            self._fc = torch.nn.Linear(input_dim, output_dim, bias=use_bias)
        else:
            self._fc = torch.nn.Linear(hidden_dims[-1], output_dim, bias=use_bias)

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        output = inputs
        for layer, activation, dropout in zip(
            self._linear_layers, self._activations, self._dropout
        ):
            output = dropout(activation(layer(output)))
        output = self._fc(output)
        return output


class Processer:
    # only part of the special tokens (valuable to classification) are contained
    special_tokens = ["ERRORTAG", "DOCUMENTTAG", "APITAG", "CODETAG", "ISSUETAG", "URLTAG", "MENTIONTAG", "NUMBERTAG", "PATHTAG", "FILETAG", "IMAGETAG"]
    inf = float("inf")
    feature_order = {"total": 0, "questioner": 1, "first": -1}
    convertion_dia = dict()
    for r in ["deeplearning4j", "spring-boot", "angular"]:
        convertion_dia[r] = json.load(open(f"./F2Chat/data/{r}_feastat.json", 'r', encoding="UTF-8"))
    convertion_mess = dict()

    def __init__(self, tokenizer, max_messages, max_tokens, merge, text_only, use_numeric=False, use_PTM=False, processed=False):
        self.max_messages = max_messages
        self.max_tokens = max_tokens  # use PTM: max length of merged sentences (the one before merge is controled by tokenizer). not use PTM: max length of sentence before merge
        self.merge = merge
        self.max_merged_tokens = max_tokens if use_PTM else 128  # max length of merged sentences
        self.text_only = True
        if not text_only and not processed:
            # need non-textual features & non-textual features need to be generated
            self.text_only = False
        self.tokenizer = tokenizer
        self.use_PTM = use_PTM
        self.use_numeric = use_numeric

    def process(self, dialogue):
        dialogue["mess_text"] = []
        if not self.text_only:
            # message_level_feature list
            # dialogue["mess_fea"] = {}
            # dialogue["mess_fea"]["spk"] = []

            # dialogue_level_features value
            # rool for naming the feature: A_B_C A-the type of feature(num:absolute value, ratio:raletive value), B-name of the feature, C-aspect of the feature
            dialogue["dia_fea"] = {}
            dialogue["dia_fea"]["num_timegap_total"] = None
            dialogue["dia_fea"]["num_timegapMean_total"] = None
            dialogue["dia_fea"]["num_timegapStd_total"] = None
            dialogue["dia_fea"]["num_blockgap_total"] = None  # number of intermediate messages
            dialogue["dia_fea"]["num_blockgapMean_total"] = None
            dialogue["dia_fea"]["num_blockgapStd_total"] = None
            dialogue["dia_fea"]["num_speaker_total"] = None
            dialogue["dia_fea"]["num_round_total"] = None
            dialogue["dia_fea"]["num_token_total"] = 0
            dialogue["dia_fea"]["num_character_total"] = 0
            dialogue["dia_fea"]["ratio_timegap_total"] = 0  # largest timegap / total timegap
            dialogue["dia_fea"]["ratio_blockgap_total"] = 0  # largest blockgap / total blockgap
            for tag in self.special_tokens:
                dialogue["dia_fea"][f"num_{tag}_total"] = 0
            time_gaps = []
            block_gaps = []

            # features related to the first message
            dialogue["dia_fea"]["num_token_first"] = None
            dialogue["dia_fea"]["num_character_first"] = None
            dialogue["dia_fea"]["num_timegap_first"] = None
            dialogue["dia_fea"]["num_blockgap_first"] = None
            dialogue["dia_fea"]["ratio_token_first"] = None
            dialogue["dia_fea"]["ratio_character_first"] = None
            dialogue["dia_fea"]["ratio_timegap_first"] = None
            dialogue["dia_fea"]["ratio_blockgap_first"] = None
            for tag in self.special_tokens:
                dialogue["dia_fea"][f"num_{tag}_first"] = None
            dialogue["dia_fea"]["ratio_questionmark_first"] = None  # bool 0-not have 1-have
            dialogue["dia_fea"]["ratio_greeting_first"] = None  # bool 0-not have 1-have

            # features related to the questioner
            dialogue["dia_fea"]["num_round_questioner"] = 0
            dialogue["dia_fea"]["num_token_questioner"] = 0
            dialogue["dia_fea"]["num_character_questioner"] = 0
            dialogue["dia_fea"]["ratio_round_questioner"] = None
            dialogue["dia_fea"]["ratio_token_questioner"] = None
            dialogue["dia_fea"]["ratio_character_questioner"] = None
            for tag in self.special_tokens:
                dialogue["dia_fea"][f"num_{tag}_questioner"] = 0

        # first_spk = dialogue["messages"][0]["user"]
        # last_spk = None
        spks = []
        
        for idx, mess in enumerate(dialogue["messages"]):
            text = replace_tokens_tags(mess["text"])  # replace special tokens
            if not self.text_only:
                mess["time"] = datetime.strptime(mess["time"], "%Y-%m-%d %H:%M:%S")
                tokens = [_ for _ in text.split(" ")]
                length_token = len(tokens)  # token_num is irrelavent with specific tokenizer (all the non-textual features should not be affected by the textual part)
                length_character = len(text)
                dialogue["dia_fea"]["num_token_total"] += length_token
                dialogue["dia_fea"]["num_character_total"] += length_character

                for t in tokens:
                    if t in self.special_tokens: 
                        dialogue["dia_fea"][f"num_{t}_total"] += 1
                
                if len(spks) == 0 or mess["user"] == spks[0]:
                    # messages from the questioner
                    dialogue["dia_fea"]["num_token_questioner"] += length_token
                    dialogue["dia_fea"]["num_character_questioner"] += length_character
                    for t in tokens:
                        if t in self.special_tokens:
                            dialogue["dia_fea"][f"num_{t}_questioner"] += 1
                
                    if len(spks) < 2:
                        # first message
                        if not dialogue["dia_fea"]["ratio_questionmark_first"]:
                            if re.search(r'\?|(what|how|why|when|which|(is|are)\s(it|there)|(are|do)\s(you|we)|(can|could)\s(i|you))[^a-z]', mess["text"], re.I):
                                dialogue["dia_fea"]["ratio_questionmark_first"] = 1
                        if not dialogue["dia_fea"]["ratio_greeting_first"]:
                            if re.search(r'(hi|hey|hello|good\s(morning|evening)|^guys)[^a-z]', mess["text"], re.I):
                                dialogue["dia_fea"]["ratio_greeting_first"] = 1

            # if last_spk is None or mess["user"] != last_spk:
                # last_spk = mess["user"]
            if len(spks) == 0 or mess["user"] != spks[-1]:
                spks.append(mess["user"])
                if not self.text_only:
                    # dialogue_level_feature
                    if idx != 0:
                        time_gaps.append((mess["time"] - dialogue["messages"][idx - 1]["time"]).total_seconds())
                        block_gaps.append(mess["index"] - dialogue["messages"][idx - 1]["index"])
                    if len(spks) == 2:
                        dialogue["dia_fea"]["num_token_first"] = dialogue["dia_fea"]["num_token_questioner"]
                        dialogue["dia_fea"]["num_character_first"] = dialogue["dia_fea"]["num_character_questioner"]
                        for t in self.special_tokens:
                            dialogue["dia_fea"][f"num_{t}_first"] = dialogue["dia_fea"][f"num_{t}_questioner"] 

                if idx >= self.max_messages:
                    # only include the first max_messages messages
                    continue
                
                dialogue["mess_text"].append(self.tokenizer.tokenize(text)[:self.max_tokens])
                if not self.text_only:
                    # message_level_feature
                    pass 
                    # dialogue["mess_fea"]["spk"].append(mess["user"] == first_spk)

            else:
                if not self.text_only:
                    pass

                if idx >= self.max_messages:
                    continue

                if self.merge:
                    dialogue["mess_text"][-1].extend(self.tokenizer.tokenize(text)[:self.max_tokens])
                    if not self.text_only:
                        pass
                else:
                    dialogue["mess_text"].append(self.tokenizer.tokenize(text)[:self.max_tokens])
                    if not self.text_only:
                        pass
                        # dialogue["mess_fea"]["spk"].append(mess["user"] == first_spk)
        if self.use_PTM:
            # need to add tokens for BERT [CLS]
            for idx, line in enumerate(dialogue["mess_text"]):
                dialogue["mess_text"][idx] = self.tokenizer.add_special_tokens(line[:self.max_merged_tokens])
        elif self.merge:
            # if noe use PTM, we should constrain the max length of merged sentences for a fair comparison
            for idx, line in enumerate(dialogue["mess_text"]):
                dialogue["mess_text"][idx] = line[:self.max_merged_tokens]

        if not self.text_only:
            total_timegap_seconds = (dialogue["messages"][-1]["time"] - dialogue["messages"][0]["time"]).total_seconds()
            dialogue["dia_fea"]["num_timegap_total"] = round(total_timegap_seconds / 60)  # min
            dialogue["dia_fea"]["num_timegapMean_total"] = round(np.mean(time_gaps) / 60) if len(time_gaps) > 0 else 0
            dialogue["dia_fea"]["num_timegapStd_total"] = round(np.std(time_gaps) / 60) if len(time_gaps) > 0 else 0
            dialogue["dia_fea"]["num_blockgap_total"] = dialogue["messages"][-1]["index"]
            dialogue["dia_fea"]["num_blockgapMean_total"] = round(np.mean(block_gaps)) if len(block_gaps) > 0 else 0
            dialogue["dia_fea"]["num_blockgapStd_total"] = round(np.std(block_gaps)) if len(block_gaps) > 0 else 0
            dialogue["dia_fea"]["num_speaker_total"] = len(set(spks))
            dialogue["dia_fea"]["num_round_total"] = len(spks)  # >=1
            dialogue["dia_fea"]["ratio_timegap_total"] = max(time_gaps) / total_timegap_seconds if len(time_gaps) > 0 else 0
            dialogue["dia_fea"]["ratio_blockgap_total"] = max(block_gaps) / dialogue["dia_fea"]["num_blockgap_total"] if len(block_gaps) > 0 else 0

            # dialogue["dia_fea"]["num_timegap_first"] = time_gaps[0] if len(time_gaps) > 0 else 0
            dialogue["dia_fea"]["num_timegap_first"] = round(time_gaps[0] / 60) if len(time_gaps) > 0 else 0
            dialogue["dia_fea"]["num_blockgap_first"] = block_gaps[0] if len(block_gaps) > 0 else 0
            dialogue["dia_fea"]["num_token_first"] = dialogue["dia_fea"]["num_token_first"] or dialogue["dia_fea"]["num_token_questioner"]
            dialogue["dia_fea"]["num_character_first"] = dialogue["dia_fea"]["num_character_first"] or dialogue["dia_fea"]["num_character_questioner"]
            dialogue["dia_fea"]["ratio_token_first"] = dialogue["dia_fea"]["num_token_first"] / dialogue["dia_fea"]["num_token_total"]
            dialogue["dia_fea"]["ratio_character_first"] = dialogue["dia_fea"]["num_character_first"] / dialogue["dia_fea"]["num_character_total"]
            # dialogue["dia_fea"]["ratio_timegap_first"] = dialogue["dia_fea"]["num_timegap_first"] / total_timegap_seconds if len(time_gaps) > 0 else 0
            dialogue["dia_fea"]["ratio_timegap_first"] = time_gaps[0] / total_timegap_seconds if len(time_gaps) > 0 else 0
            dialogue["dia_fea"]["ratio_blockgap_first"] = dialogue["dia_fea"]["num_blockgap_first"] / dialogue["dia_fea"]["num_blockgap_total"] if len(block_gaps) > 0 else 0
            dialogue["dia_fea"]["ratio_questionmark_first"] = dialogue["dia_fea"]["ratio_questionmark_first"] or 0
            dialogue["dia_fea"]["ratio_greeting_first"] = dialogue["dia_fea"]["ratio_greeting_first"] or 0
            if len(spks) == 1:
                for t in self.special_tokens:
                    dialogue["dia_fea"][f"num_{t}_first"] = dialogue["dia_fea"][f"num_{t}_questioner"]

            dialogue["dia_fea"]["num_round_questioner"] = spks.count(spks[0])
            dialogue["dia_fea"]["ratio_round_questioner"] = dialogue["dia_fea"]["num_round_questioner"] / dialogue["dia_fea"]["num_round_total"]
            dialogue["dia_fea"]["ratio_token_questioner"] = dialogue["dia_fea"]["num_token_questioner"] / dialogue["dia_fea"]["num_token_total"]
            dialogue["dia_fea"]["ratio_character_questioner"] = dialogue["dia_fea"]["num_character_questioner"] / dialogue["dia_fea"]["num_character_total"]

            room = dialogue["project"]  # the corresponding chatroom
            if self.use_numeric:
                eps = 1e-1
                for k, v in dialogue["dia_fea"].items():
                    if "ratio" in k:
                        # float between [0,1], do not need normalize
                        pass
                    else:
                        # num feature
                        # Processer.dia_fea[k]["v"].append(log(v + eps, 10))
                        _, feature, type_ = k.split('_')
                        v = min(v, Processer.convertion_dia[room][feature]["max"][Processer.feature_order[type_]])  # cut to the maximum
                        v = v if not Processer.convertion_dia[room][feature]["use_log"] else log_(v, base=10)
                        dialogue["dia_fea"][k] = v
                return dialogue
            
            '''
            # not implemented
            # turn numeriacal values into str
            for k, v in dialogue["dia_fea"].items():
                if not Processer.convertion_dia[room][k]["use_log"]:
                    for idx, thres in enumerate(Processer.convertion_dia[room][k]["thres"]):
                        if v <= thres:
                            dialogue["dia_fea"][k] = f"{idx}"
                            break
                else:
                    dialogue["dia_fea"][k] = "%.1f" % log(v + eps, 10)
            '''

        return dialogue


def log_(x, base=10):
    return log(x, base) if x != 0 else 0


def replace_tokens_baseline(content):
    # special tokens replacement in FRMiner(for Slack)
    # those we hide are not applicable to our dataset(Gitter)
    # we also adapt/add some replacements to our dataset

    # content = re.sub(r"\*\*I'm submitting a.+?\\r\\n\\r\\n\*\*", "", content)  # not applicable
    content = re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', 'URL ', content)  # reserve
    content = re.sub(r'[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}', 'EMAIL ', content)  # reserve
    content = re.sub(r'@\S+', 'MEN ', content)  # add
    content = re.sub(r'[(0-9)+(a-z)+]{10,}', 'HASH_ID ', content)  # reserve
    content = re.sub(r'#\d+\s', 'PR_ID ', content)  # reserve
    content = re.sub(r'`.*?`', 'SCODE ', content, re.S)  # adapt (short code)
    content = re.sub(r"```.*?```", 'LCODE ', content, re.S)  # adapt (long code)
    content = re.sub(r'<[^>]*>|<\/[^>]*>', 'HTML ', content)  # reserve
    # content = re.sub(r'-\s\[\s*x?\s*\]\s((feature\srequest)|(bug\sreport)|other)', '', content)  # not applicable
    return content


# fine-grained special tokens replacement (F2Chat)
DOC_PATTERN_URL = re.compile(r"example|tutorial|doc|api|quickstart|note|guide|blog|using|reference|\.io|sample|demo|template", re.I)
DOC_PATTERN_CODE = re.compile(r"example", re.I)
ERROR_PATTERN = re.compile(r"exception|error|warning|404|can't|can\s{0,1}not|could\s{0,1}not|undefined", re.I)
ISSUE_PATTERN = re.compile(r"issue|pull|#[0-9]+|[0-9]{4,}", re.I)
TAG_PATTERN = re.compile(r'CODETAG|APITAG')
CODE_PATTERN = re.compile(r'[=;%$/<>\{\}\[\]]|public\sstatic\s(void){0,1}(\smain){0,1}|String')
# match from begin to end
NUM_PATTERN = re.compile(r'^(([^a-uwyz]+?\d[^a-uwyz]*(beta[0-9]+){0,1})|(\s*beta[0-9]+))\s*$', re.I)
PATH_PATTERN = re.compile(r'^\s*([^\s\(\)]+?[/\\]){2,}[^\s\(\)]*\s*$')
API_PATTERN = re.compile(r'^\s*\S+\s*$')

def replace_tokens_tags(content):
    # the tail "tag" is to make sure to differentiate tag and ordinary word under un-cased situation 
    # API: API short code || CODE: long code
    for _ in re.finditer(r"```.*?```", content, flags=re.S):
        code = _.group()
        if ERROR_PATTERN.search(code):
            content = content.replace(code, " ERRORTAG ", 1)
        elif PATH_PATTERN.search(code[3:-3]):
            content = content.replace(code, " PATHTAG ", 1)
        elif NUM_PATTERN.search(code[3:-3]):
            content = content.replace(code, " NUMBERTAG ", 1)
        elif API_PATTERN.search(code[3:-3]):
            if DOC_PATTERN_CODE.search(code):
                content = content.replace(code, " DOCUMENTTAG ", 1)
            else:
                content = content.replace(code, " APITAG ", 1)
        else:
            content = content.replace(code, " CODETAG ", 1)
        '''
        else:
            content = content.replace(code, " CODETAG ", 1)
        '''

    for _ in re.finditer(r'`.*?`', content, flags=re.S):
        code = _.group()
        if ERROR_PATTERN.search(code):
            content = content.replace(code, " ERRORTAG ", 1)
        elif PATH_PATTERN.search(code[1:-1]):
            content = content.replace(code, " PATHTAG ", 1)
        elif NUM_PATTERN.search(code[1:-1]):
            content = content.replace(code, " NUMBERTAG ", 1)
        elif API_PATTERN.search(code[1:-1]):
            if DOC_PATTERN_CODE.search(code):
                content = content.replace(code, " DOCUMENTTAG ", 1)
            else:
                content = content.replace(code, " APITAG ", 1)
        else:
            content = content.replace(code, " CODETAG ", 1)
        '''
        else:
            content = content.replace(code, " CODETAG ", 1)
        '''
    
    content = re.sub(r'<[^>]*>{3,}', ' CODETAG ', content)
    content = re.sub(r'<[^>]*?\s[^>]*>', ' CODETAG ', content)
    
    for _ in re.finditer(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content):
        url = _.group()
        if re.search(r'gist|stackblitz', url, flags=re.I):
            content = content.replace(url, " CODETAG ", 1)  # gist and stackblitz are used for long code snippets or stacktraces 
        elif re.search(r'files\.gitter', url, flags=re.I):
            content = content.replace(url, " IMAGETAG ", 1)
        elif DOC_PATTERN_URL.search(url):
            content = content.replace(url, " DOCUMENTTAG ", 1)
        elif ISSUE_PATTERN.search(url):
            content = content.replace(url, " ISSUETAG ", 1)
        else:
            content = content.replace(url, " URLTAG ", 1)

    content = re.sub(r'\[.*\.[A-Za-z]{2,4}\]([^A-Za-z0-9]*IMAGETAG)*', ' IMAGETAG ', content)
    
    # content = re.sub(r'[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}', ' EMAILTAG ', content)  # Email
    # replace mention, make sure that code like "@ExceptionHandler" not replaced as MENTIONTAG
    content = re.sub(r'^\s*@\S+|@([^A-Z\s]\S+|\S*?[0-9_\-]\S*|Toxicable|DoraRajappan|RigaNik|EreckGordon|RoboZoom|DanielNetzer|OrangeDog|TomRauchenwald|AlexCzar|IEnoobong|Redmancometh|BenEfrati|Bakuryu|VicPopescu|SAGOlab|DaveMBush|BuntyBru|AlexDBlack|RobAltena|KolyaIvankov|SidneyLann|Tschigger|SanderGielisse|MilanVanca|MrDallOca)', ' MENTIONTAG ', content)

    content = re.sub(r'\S+?example\S*', ' DOCUMENTTAG ', content, re.I)  # iteratorExample.java dl4j-examples

    content = re.sub(r'\S+?(Error|Exception)([^A-Za-z\s]\S*|\s|$)|404', ' ERRORTAG ', content)  # java.lang.IllegalStateException or 404

    content = re.sub(r'#[0-9]+', ' ISSUETAG ', content)  # issue

    # content = re.sub(r'\s[A-Za-z]+\.(ml|xml|png|csv|jar|sh|sbt|zip|exe|md|txt|js|yml|yaml|json|sql|html|jsp|php|prod|scss|ts)[,\.]*\s+', ' FILETAG ', content, re.I)

    content = re.sub(r'([^\s\(\)]+?[/\\]){2,}[^\s\(\)]*', ' PATHTAG ', content)  # PATHTAG

    for _ in re.finditer(r'\s(\S+?\.(ml|xml|png|csv|jar|sh|sbt|zip|exe|md|txt|js|yml|yaml|json|sql|html|jsp|php|prod|scss|ts))[?,\.]{0,1}\s', content):
        # do not eliminate \r \n
        content = content.replace(_.group(1), " FILETAG ", 1)

    content = re.sub(r'\S{17,}', ' APITAG ', content)

    # content = re.sub(r'\s(deeplearning4j|nd4j|dl4j|angular|angularjs|angular[\s\-]{0,1}cli|cli|spring|spring[\s\-]{0,1}boot|webstorm|redis)[,\.]{0,1}\s', ' LIBRARYTAG ', content, flags=re.I)  # domain specific
    for _ in re.finditer(r'\s(deeplearning4j|nd4j|dl4j|angular|angularjs|angular[\s\-]{0,1}cli|cli|spring[\s\-]{0,1}(boot){0,1}|webstorm|redis|mongodb)[?,\.]{0,1}\s', content, flags=re.I):
        # do not eliminate \r \n
        content = content.replace(_.group(1), ' LIBRARYTAG ', 1)

    content = re.sub(r'\S+?((\(\))|(\[\]))\S*|[^,;\.\s]{3,}?\.\S{4,}|\S+?([a-z][A-Z]|[A-Z][a-z]{2,}?)\S*|@\S+|<\S*?>', ' APITAG ', content)  # contain () / contain . / camel expertDeveloper / @ExceptionHandler

    content = re.sub(r'[^a-uwyz]+?\d[^a-uwyz]*(beta[0-9]+){0,1}|beta[0-9]+', ' NUMBERTAG ', content, flags=re.I)

    content = re.sub(r'public\sstatic\s(void){0,1}(\smain){0,1}|String', ' CODETAG ', content)

    # replace code that are not correctly formated (using ```...```)
    content = re.sub(r'\r\n', '\r\n\r\n', content)
    for _ in re.finditer(r'\r\n.+?\r\n', content, re.S):
        code = _.group()
        # if CODE_PATTERN.search(code):
        if len(CODE_PATTERN.findall(code)) >= 3:
            if "ERRORTAG" in code or ERROR_PATTERN.search(code):
                content = content.replace(code, " ERRORTAG ", 1)
            else:
                content = content.replace(code, " CODETAG ", 1)
        elif len(re.findall(r'ERRORTAG|PATHTAG|NUMBERTAG|CODETAG|APITAG', code)) >= 2:
            if "ERRORTAG" in code or ERROR_PATTERN.search(code):
                content = content.replace(code, " ERRORTAG ", 1)
            elif TAG_PATTERN.search(code):
                content = content.replace(code, " CODETAG ", 1)
    while True:
        _ = re.search(r'(ERRORTAG|PATHTAG|NUMBERTAG|CODETAG|APITAG)(.*?[=;%$/<>\{\}\[\]]){2,}?.*?(ERRORTAG|PATHTAG|NUMBERTAG|CODETAG|APITAG)', content, re.S)
        if _ is None:
            break
        code = _.group()
        if "ERRORTAG" in code or ERROR_PATTERN.search(code):
            content = content.replace(code, " ERRORTAG ", 1)
        else:
            content = content.replace(code, " CODETAG ", 1)
    for _ in re.finditer(r'[^a-zA-Z0-9]*(((ERRORTAG|PATHTAG|NUMBERTAG|CODETAG|APITAG)([a-zA-Z0-9]{0,3}[^a-zA-Z0-9])*){4,}|((ERRORTAG|PATHTAG|NUMBERTAG|CODETAG|APITAG)[^a-zA-Z0-9]*){2,})', content):
        code = _.group()
        if "ERRORTAG" in code or ERROR_PATTERN.search(code):
            content = content.replace(code, " ERRORTAG ", 1)
        elif TAG_PATTERN.search(code):
            content = content.replace(code, " CODETAG ", 1)

    content = re.sub(r'^\s*>\s*.*?(\r\n|\n\n|\n|$)', ' ', content)

    content = re.sub(r'[\r\n\t]', ' ', content)

    # remove extra spaces
    content = ' '.join([_ for _ in content.split(' ') if _ != ''])
    return content


def pack2sequence(seq1, mask1, seq2, mask2):
    seq1_lens = get_lengths_from_binary_sequence_mask(mask1)
    seq2_lens = get_lengths_from_binary_sequence_mask(mask2)
    combined_lens = seq1_lens + seq2_lens
    max_len, _ = torch.max(combined_lens + torch.tensor([5], device=seq1.device), dim=0)
    combined_tensor = torch.zeros(combined_lens.size()[-1], max_len, seq1.size()[-1], device=seq1.device)
    # print(combined_tensor.size())
    # print(combined_lens)
    for i, (len1, len2) in enumerate(zip(seq1_lens, seq2_lens)):
        combined_tensor[i, :len1, :] = seq1[i, :len1, :]
        combined_tensor[i, len1:len1 + len2, :] = seq2[i, :len2, :]
    combined_mask = get_mask_from_sequence_lengths(combined_lens, max_len)
    sorted_tensor, _, restoration_indices, permutation_index = sort_batch_by_length(
        combined_tensor, combined_lens)
    combined_mask = combined_mask.index_select(0, permutation_index)
    return sorted_tensor, combined_mask, restoration_indices, permutation_index


def generate_class_weights(room, fold, archived_vocab, namespace="label_tags"):
    # generate class weights to tackle the unbalanced dataset problem
    archived_path = PATH + archived_vocab
    vocab = Vocabulary.from_files(archived_path)
    idx2token = vocab.get_index_to_token_vocabulary(namespace)

    dialogues = json.load(open(PATH + f"/data/{room}_train_{fold}.json", 'r', encoding="UTF-8"))
    dia_num = {}
    for dia in dialogues:
        intention = dia["intention"]
        if dia_num.get(intention) is None:
            dia_num[intention] = 0
        dia_num[intention] += 1
    
    class_num = len(list(dia_num.keys()))
    k = np.sum(list(dia_num.values())) / class_num
    class_weights = {}

    if namespace == "label_tags":
        assert len(list(idx2token.keys())) == np.square(class_num)

        for idx, label_tag in idx2token.items():
            label1, label2 = label_tag.split("@")
            if label1 == label2:
                class_weights[label_tag] = k / dia_num[label1]
            else:
                class_weights[label_tag] = 1
    else:
        assert len(list(idx2token.keys())) == class_num

        for idx, label in idx2token.items():
            class_weights[label] = k / dia_num[label]

    print(class_weights)
    with open(PATH + f"/data/{room}_weights_{fold}_stage{1 if namespace == 'label_tags' else 2}.json", 'w', encoding="UTF-8") as f:
        json.dump(class_weights, f)


def make_golden(room, fold, golden_num=5):
    # randomly sample golden samples from the train set(used for testing of the F2Chat-t)
    dialogues = json.load(open(PATH + f"/data/{room}_train_{fold}.json", 'r', encoding="UTF-8"))
    all_data = {}
    for dia in dialogues:
        intention = MAPPING[dia["intention"]]
        if all_data.get(intention) == None:
            all_data[intention] = []
        all_data[intention].append(dia)
    
    all_data_num = {}
    for key, value in all_data.items():
        all_data_num[key] = len(value)

    sample_num = min(list(all_data_num.values()))
    # print(sample_num)
    sample_num = golden_num
    golden_dialogues = []
    for key, value in all_data.items():
        selected_indices = random.sample(range(all_data_num[key]), sample_num)
        for index, dia in enumerate(value):
            if index in selected_indices:
                golden_dialogues.append(dia)

    with open(PATH + f"/data/{room}_golden_{fold}.json", "w", encoding="UTF-8") as f:
        json.dump(golden_dialogues, f, indent=4)


def divide_dataset(room, fold_num=5):
    # divide the dataset into train sets and test sets using stratified random sampling (retain the distribution in the original dataset)
    dialogues = json.load(open(PATH + f"/data/{room}.json", 'r', encoding="UTF-8"))
    class_distribution = {}
    for dia in dialogues:
        intention = dia["intention"]
        if class_distribution.get(intention) == None:
            class_distribution[intention] = []
        class_distribution[intention].append(dia)

    class_num = {}
    availabel_indices = {}
    for key, value in class_distribution.items():
        num = len(value)
        class_num[key] = {"base": int(num / fold_num), "addition": num % fold_num}
        availabel_indices[key] = range(num)
    
    print(class_num)

    for i in range(fold_num):
        # print(f"============={i}=================")
        testset = []
        trainset = []
        for key, value in class_num.items():
            if (i + 1) != fold_num:
                if i < value["addition"]:
                    test_indices = random.sample(availabel_indices[key], value["base"] + 1)
                else:
                    test_indices = random.sample(availabel_indices[key], value["base"])
            else:
                test_indices = availabel_indices[key]  # last folder don't need to select
            for index, sample in enumerate(class_distribution[key]):
                if index in test_indices:
                    testset.append(sample)
                else:
                    trainset.append(sample)
        
            # update available_indices
            # print(key, len(availabel_indices[key]), len(test_indices))
            availabel_indices[key] = list(set(availabel_indices[key]) - set(test_indices))

        with open(PATH + f"/data/{room}_test_{i}.json", "w", encoding="UTF-8") as f:
            json.dump(testset, f, indent=4)
        with open(PATH + f"/data/{room}_train_{i}.json", "w", encoding="UTF-8") as f:
            json.dump(trainset, f, indent=4)


def build_dataset_cross(room):
    # use leave-one-project-out-cross-validation for cross-poject testing scenario
    rooms = ["deeplearning4j", "angular", "spring-boot"]
    train_dialogues = list()
    test_dialogues = list()
    for r in rooms:
        dias = json.load(open(PATH + f"/data/{r}.json", 'r', encoding="UTF-8"))
        if r == room:
            test_dialogues.extend(dias)
        else:     
            train_dialogues.extend(dias)
    
    print("train:", len(train_dialogues))
    print("test:", len(test_dialogues))
    with open(PATH + f"/data/{room}_test_cross.json", 'w') as f:
        json.dump(test_dialogues, f, indent=4)
    with open(PATH + f"/data/{room}_train_cross.json", 'w') as f:
        json.dump(train_dialogues, f, indent=4)


def divide_trainset(room, fold, fold_num=5):
    # further divide the trainset into train sets and validation sets
    # same to divide_dataset(), also using stratified random sampling
    dialogues = json.load(open(PATH + f"/data/{room}_train_{fold}.json", 'r', encoding="UTF-8"))
    class_distribution = {}
    for dia in dialogues:
        intention = dia["intention"]
        if class_distribution.get(intention) == None:
            class_distribution[intention] = []
        class_distribution[intention].append(dia)

    class_num = {}
    availabel_indices = {}
    for key, value in class_distribution.items():
        num = len(value)
        class_num[key] = {"base": int(num / fold_num), "addition": num % fold_num}
        availabel_indices[key] = range(num)
    
    print(class_num)

    for i in range(fold_num):
        # print(f"============={i}=================")
        testset = []
        trainset = []
        for key, value in class_num.items():
            if (i + 1) != fold_num:
                if i < value["addition"]:
                    test_indices = random.sample(availabel_indices[key], value["base"] + 1)
                else:
                    test_indices = random.sample(availabel_indices[key], value["base"])
            else:
                test_indices = availabel_indices[key]  # last fold don't need to select
            for index, sample in enumerate(class_distribution[key]):
                if index in test_indices:
                    testset.append(sample)
                else:
                    trainset.append(sample)
        
            # update available_indices
            # print(key, len(availabel_indices[key]), len(test_indices))
            availabel_indices[key] = list(set(availabel_indices[key]) - set(test_indices))

        with open(PATH + f"/data/{room}_validation_{fold}_v{i}.json", "w", encoding="UTF-8") as f:
            json.dump(testset, f, indent=4)
        with open(PATH + f"/data/{room}_train_{fold}_v{i}.json", "w", encoding="UTF-8") as f:
            json.dump(trainset, f, indent=4)


def count_shallow_feature(room):
    # calculate the non-textual features and count the max&min for each non-textual feature
    dialogues = json.load(open(PATH + f"/data/{room}.json", 'r', encoding="UTF-8"))
    processer = Processer(tokenizer=WhitespaceTokenizer(), max_messages=20, max_tokens=10, merge=True, text_only=False, use_numeric=True, use_PTM=False)
    dialogues = [processer.process(_) for _ in dialogues]
    stat = dict()

    x = range(len(dialogues))  # x-axis
    c = list()  # color
    has_color = False
    intentions = None

    dialogue_level_feature = list(dialogues[0]["dia_fea"].keys())
    # message_level_feature = list(dialogues[0]["mess_fea"].keys())
    for fea in dialogue_level_feature:
        feature_dict = {}

        for dia in dialogues:
            intention = dia["intention"]
            if intention not in feature_dict:
                feature_dict[intention] = list()
            feature_dict[intention].append(dia["dia_fea"][fea])

        if intentions is None:
            intentions = list(feature_dict.keys())
        y = []
        for intent, fea_list in feature_dict.items():
            y.extend(fea_list)
            if not has_color:
                c.extend([intentions.index(intent)]*len(fea_list))
        has_color = True

        max_ = max(y)
        min_ = min(y)
        print(fea, max_, min_)
        base = max_ - min_
        stat[fea] = {"use": True, "embedding_dim": 5, "max": max_, "min": min_}
        if "ratio" not in fea:
            y = [(_ - min_) / base for _ in y]
            
        plt.figure()
        plt.scatter(x, y, s=1, c=c)
        # plt.ylim((0, 5000))
        plt.savefig(PATH + f"/pics/{room}/{fea}.png", dpi=600)
        plt.close()

    with open(PATH + f"/data/{room}_minmax.json", 'w') as f:
        json.dump(stat, f, indent=4)


def add_nontextual_feature(file):
    # generate non-textual feautures for file, the non-textual features are prepared before training
    dialogues = json.load(open(PATH + f"/data/{file}.json", 'r', encoding="UTF-8"))
    dialogues_updated = copy.deepcopy(dialogues)
    processer = Processer(tokenizer=WhitespaceTokenizer(), max_messages=20, max_tokens=10, merge=True, text_only=False, use_numeric=True, use_PTM=False)
    dialogues = [processer.process(_) for _ in dialogues]
    fea_minmax = dict()
    for room in ["deeplearning4j", "spring-boot", "angular"]:
        fea_minmax[room] = json.load(open(PATH + f"/data/{room}_minmax.json", 'r', encoding="UTF-8"))

    for dia in dialogues:
        room = dia["project"]
        for fea, v in dia["dia_fea"].items():
            if "ratio" in fea:
                continue
            dia["dia_fea"][fea] = (v - fea_minmax[room][fea]["min"]) / (fea_minmax[room][fea]["max"] - fea_minmax[room][fea]["min"])
           
    for idx, dia in enumerate(dialogues):
        dialogues_updated[idx]["dia_fea"] = dia["dia_fea"]

    with open(PATH + f"/data/{file}_1.json", 'w') as f:
        json.dump(dialogues_updated, f, indent=4)


PATH = "./F2Chat_s"
# use the first mapping, we can evaluate the performance on level 1(hierarchical taxonomy)
# level 1
MAPPING = {'programming_problem': 'problem_report', 'library_problem': 'problem_report', 'documentation_problem': 'problem_report',
            'programming_information': 'information_retrieval', 'library_information': 'information_retrieval', 'documentation_information': 'information_retrieval', 'general_information': 'information_retrieval',
            'technical_discussion': 'project_management', 'task_progress': 'project_management'}

# level 2
MAPPING = {'programming_problem': 'programming_problem', 'library_problem': 'library_problem', 'documentation_problem': 'documentation_problem',
            'programming_information': 'programming_information', 'library_information': 'library_information', 'documentation_information': 'documentation_information', 'general_information': 'general_information',
            'technical_discussion': 'technical_discussion', 'task_progress': 'task_progress'}


if __name__ == "__main__":
    room = "deeplearning4j"
    fold = 0
    # divide_dataset(room)
    # divide_trainset(room="deeplearning4j", fold=0)
    # make_golden(room="deeplearning4j", fold=0, golden_num=5)
    # "google/bert_uncased_L-4_H-512_A-8"
    # build_dataset_cross(room="deeplearning4j")
    # add_nontextual_feature("deeplearning4j_test_0")