from dataclasses import dataclass


@dataclass
class RatesHelper:
    true_positives: set
    true_negatives: set
    false_positives: set
    false_negatives: set

    @property
    def tpr(self):
        num_real_positives = len(self.true_positives) + len(self.false_negatives)
        return len(self.true_positives) / num_real_positives

    @property
    def fpr(self):
        num_real_negatives = len(self.false_positives) + len(self.true_negatives)
        return len(self.false_positives) / num_real_negatives



