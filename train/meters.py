import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=''):
        self.name = name
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
            val:    hit count.
            n:      size count.
        """

        self.sum += val
        self.count += n
        self.avg = float(self.sum) / self.count


class MatrixMeter(object):
    """
    the (i,j) of the matrix is the category i to be recognized as category j.
    """

    def get_recall(self, idx):
        # how many targets
        bottom = self.hit_matrix[idx].sum()
        # how many searched
        top = float(self.hit_matrix[idx, idx])
        return top / bottom if bottom != 0 else 0

    def get_precision(self, idx):
        # how many predictions
        bottom = self.hit_matrix[:, idx].sum()
        # how many correct
        top = float(self.hit_matrix[idx, idx])
        return top / bottom if bottom != 0 else 0

    def get_f1score(self, idx):
        r = self.get_recall(idx)
        p = self.get_precision(idx)
        if (p + r) == 0:
            return 0
        return 2 * p * r / (p + r)

    def __init__(self, classes, default_metric='acc'):
        self.classes = classes
        self.default_metric = default_metric
        self.hit_matrix = np.zeros((len(self.classes), len(self.classes)))

    def update(self, pred, target):
        for i in range(len(target)):
            self.hit_matrix[target[i], pred.argmax(1)[i]] += 1

    def __str__(self):
        if self.default_metric == 'acc':
            return f'Acc: >>[{self.acc():3.1%}]<< .'
        elif self.default_metric == 'f1score':
            return f'F1score: >>[{self.get_f1score(1):3.1%}]<< .'
        elif self.default_metric == 'pr1':
            return f'PR of {self.classes[1]}: >>[{self.get_precision(1):3.1%}]<< .'
        else:
            raise NotImplementedError

    def get_main(self):
        if self.default_metric == 'acc':
            return self.acc()
        elif self.default_metric == 'f1score':
            return self.get_f1score(1)
        elif self.default_metric == 'pr1':
            return self.get_precision(1)
        else:
            raise NotImplementedError

    def acc(self):
        return self.hit_matrix.diagonal().sum() / self.hit_matrix.sum()

    def get_matrix(self):
        return self.hit_matrix / self.hit_matrix.sum(1).reshape(-1, 1)

    def report(self):
        s = '\n======================== Report =======================\n'
        s += f'======================== {self.get_main()} =======================\n'
        s += '\n'
        for i, c in enumerate(self.classes):
            s += f'{c:20s}:\tPR: {self.get_precision(i):5.1%},\tRR: {self.get_recall(i):5.1%},' \
                 f'\t F1: {self.get_f1score(i):5.1%}.\n'

        return s + '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'

    def get_str_f1score(self, idx):
        return f'F1-score of {self.classes[idx]}: {self.get_f1score(idx):3.1%}'
