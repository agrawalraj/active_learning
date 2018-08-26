from dataclasses import dataclass, asdict


@dataclass
class RatesHelper:
    true_positives: set
    true_negatives: set
    false_positives: set
    false_negatives: set

    @property
    def tpr(self):
        """
        Percent of actual positives labelled as positives

        AKA: sensitivity, recall
        """
        num_real_positives = len(self.true_positives) + len(self.false_negatives)
        return len(self.true_positives) / num_real_positives if num_real_positives != 0 else 1

    @property
    def fpr(self):
        """
        Percent of actual negatives labelled as positives
        """
        num_real_negatives = len(self.false_positives) + len(self.true_negatives)
        return len(self.false_positives) / num_real_negatives if num_real_negatives != 0 else 0

    @property
    def tnr(self):
        """
        Percent of actual negatives labelled as negatives
        """
        num_real_negatives = len(self.false_positives) + len(self.true_negatives)
        return len(self.true_negatives) / num_real_negatives if num_real_negatives != 0 else 1

    @property
    def fnr(self):
        """
        Percent of actual positives labelled as negatives
        """
        num_real_positives = len(self.true_positives) + len(self.false_negatives)
        return len(self.false_positives) / num_real_positives if num_real_positives != 0 else 0

    @property
    def ppv(self):
        """
        Percent of labelled positives that are actual positives

        AKA: recall
        """
        num_labelled_positives = len(self.true_positives) + len(self.false_positives)
        return len(self.true_positives) / num_labelled_positives if num_labelled_positives != 0 else 1

    @property
    def npv(self):
        """
        Percent of labelled negatives that are actual negatives

        AKA: specificity
        """
        num_labelled_negatives = len(self.true_negatives) + len(self.false_negatives)
        return len(self.true_negatives) / num_labelled_negatives if num_labelled_negatives != 0 else 1

    def to_dict(self, with_sets=False):
        d = {
            'tpr': self.tpr,
            'fpr': self.fpr,
            'tnr': self.tnr,
            'fnr': self.fnr,
            'ppv': self.ppv,
            'npv': self.npv
        }
        if with_sets:
            d = {**asdict(self), **d}
        return d


if __name__ == '__main__':
    true_positives = {1}
    true_negatives = {2}
    false_positives = {3}
    false_negatives = {4}
    rh = RatesHelper(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )

