# Wrapper for scipy continuious metrics

from typing import Any
from sklearn.metrics import roc_auc_score, average_precision_score


class ScorerRequiresContPred:
    def __init__(self, scorer, name) -> None:
        self.scorer = scorer
        self.name = name
        self.greater_is_better = True
        self.cond_weights = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.scorer(*args, **kwds)


roc_auc = ScorerRequiresContPred(roc_auc_score, 'ROC AUC')
average_precision = ScorerRequiresContPred(average_precision_score, 'Average Precision')
