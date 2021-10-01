from typing import Optional
import torch
from allennlp.training.metrics import Metric


@Metric.register("siamese_measure_v1")
class SiameseMeasureV1(Metric):
    def __init__(self, vocab) -> None:
        self._same_idx = vocab.get_token_index("same", namespace="lables")  # same_idx
        self._idx2token = vocab.get_index_to_token_vocabulary(namespace="label_tags")
        self._same_score = 0
        self._diff_score = 0
        
    def __call__(self,
                 predictions: torch.Tensor,
                 label_tags: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        diff = 0
        for probs, tags in zip(predictions.tolist(), label_tags.tolist()):
            sample, golden = self._idx2token[tags].split("@")
            score = probs[self._same_idx]
            if sample == golden:
                # every unit is started with positive pair, and some negetive ones after
                self._diff_score += diff
                diff = 0
                self._same_score += score
            else:
                if score > diff:
                    diff = score
        self._diff_score += diff

    def get_metric(self, reset: bool):
        score = self._same_score - self._diff_score
        if reset:
            self.reset()
        return score

    def reset(self) -> None:
        self._same_score = 0
        self._diff_score = 0


@Metric.register("siamese_measure_v2")
class SiameseMeasureV2(Metric):
    def __init__(self, vocab) -> None:
        self._same_idx = vocab.get_token_index("same", namespace="lables")  # same_idx
        self._idx2token = vocab.get_index_to_token_vocabulary(namespace="label_tags")
        self._num_correct = 0
        
    def __call__(self,
                 predictions: torch.Tensor,
                 label_tags: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        diff = 0
        same = 0
        for probs, tags in zip(predictions.tolist(), label_tags.tolist()):
            sample, golden = self._idx2token[tags].split("@")
            score = probs[self._same_idx]
            if sample == golden:
                # every unit is started with positive pair, and some negetive ones after
                if same > diff:
                    self._num_correct += 1
                diff = 0
                same = score
            else:
                if score > diff:
                    diff = score
        
        if same > diff:
            self._num_correct += 1

    def get_metric(self, reset: bool):
        num_correct = self._num_correct
        if reset:
            self.reset()
        return num_correct

    def reset(self) -> None:
        self._num_correct = 0