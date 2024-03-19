from collections import defaultdict
import os
import matplotlib.pyplot as plt
from scipy import spatial
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score, RocCurveDisplay, roc_auc_score, confusion_matrix

import math
import torch.nn.functional as F

from math import sqrt
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.functional.classification.auroc import binary_auroc, multiclass_auroc
from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from torchmetrics.functional.classification.specificity import multiclass_specificity
from torchmetrics.functional.classification.precision_recall import multiclass_recall
from torchmetrics.functional.classification.confusion_matrix import binary_confusion_matrix, multiclass_confusion_matrix
from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from typing_extensions import Literal
from torchmetrics.utilities.data import dim_zero_cat
from sklearn.calibration import CalibrationDisplay



class BinaryEvalMetrics(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
        **kwargs: Any,
    ) -> None:
        super(BinaryEvalMetrics, self).__init__(**kwargs)
        self.multidim_average = multidim_average
        self._create_state(size=1, multidim_average=multidim_average)
        

    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = list
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("prob", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("target", default(), dist_reduce_fx=dist_reduce_fx)
    
    def update(self, logit: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        prob = torch.sigmoid(logit.detach())
        target = target.long().detach()
        
        if self.multidim_average == "samplewise":
            self.prob.append(prob)
            self.target.append(target)
    
    def _final_state(self) -> Tuple[Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        prob = dim_zero_cat(self.prob)
        target = dim_zero_cat(self.target)
        
        return prob, target
    
    def compute(self) -> float:
        """Compute metrics based on inputs passed in to ``update`` previously."""
        prob, target = self._final_state()
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        AUC = binary_auroc(prob, target)
        
        return AUC
        
    def on_epoch_end_compute(self, best_thres: float=None) -> Dict:
        """Compute metrics based on inputs passed in to ``update`` previously."""
        
        metric_dict = {}
        
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        
        fpr, tpr, thres = binary_roc(prob, target)
        idx = torch.argmax(tpr - fpr).item()
        
        if best_thres == None:
            best_thres = thres[idx].item()
        
        # Compute AUC
        AUC = binary_auroc(prob, target)
        CI95_lower, CI95_upper = self.bootstrap_auc(prob, target)
        
        # Prediction
        pred = prob.clone()
        pred[pred<best_thres] = 0
        pred[pred>=best_thres] = 1
        
        # Compute other metrics
        a = 1
        conf_mat = binary_confusion_matrix(pred, target, best_thres)
        tn, fp, fn, tp = conf_mat[0, 0].item(), conf_mat[0, 1].item(), conf_mat[1, 0].item(), conf_mat[1, 1].item()
        
        sensitivity = tp / max((tp + fn), 1e-6)
        specificity = tn / max((fp + tn), 1e-6)
        precision = tp / max((tp + fp), 1e-6)
        accuracy = (tp + tn) / max((tp + tn + fp + fn), 1e-6)
        f1 = 2 * precision * sensitivity / max((precision + sensitivity), 1e-6)
        
        metric_dict['Sensitivity'] = float(sensitivity)
        metric_dict['Specificity'] = float(specificity)
        metric_dict['Precision'] = float(precision)
        metric_dict['Accuracy'] = float(accuracy)
        metric_dict['F1_Score'] = float(f1)
        metric_dict['AUC'] = float(AUC)
        metric_dict['CI95_lower'] = float(CI95_lower)
        metric_dict['CI95_upper'] = float(CI95_upper)
        metric_dict['Best_thres'] = float(best_thres)
        
        ECE = self.metric_ece()
        metric_dict['ECE'] = float(ECE)
        
        return metric_dict
        
    def bootstrap_auc(self, prob, target, n_bootstraps=5000, alpha=0.05) -> Tuple[float, float]:
        assert len(target) == len(prob)
        n = len(target)
        auc_scores = []

        # Bootstrap resampling
        for _ in range(n_bootstraps):
            indices = torch.randint(0, len(target), (n,))
            auc = binary_auroc(prob[indices], target[indices]).item()
            auc_scores.append(auc)

        # Compute Confidence Interval
        sorted_scores = torch.tensor(sorted(auc_scores))
        lower = torch.quantile(sorted_scores, (alpha / 2.)).item()
        upper = torch.quantile(sorted_scores, (1 - alpha / 2.)).item()

        return lower, upper
    
    def plot_graphs(self, result_root_path):
        self.plot_roc_curve(result_root_path)
        self.plot_calibration_curve(result_root_path)
        
    def plot_roc_curve(self, result_root_path):
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        
        fpr, tpr, _ = binary_roc(prob, target)
        final_auc = auc(fpr, tpr)
        
        plt.plot(fpr.numpy(), tpr.numpy(), label='ROC (area = %0.2f)' % (final_auc))
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        save_path = os.path.join(result_root_path, 'images')
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, 'ROC_curve.jpg'))
        plt.close()
        plt.clf()


    def plot_calibration_curve(self, result_root_path):
        prob, target = self._final_state()
    
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        
        display = CalibrationDisplay.from_predictions(target.numpy(), prob.numpy(), n_bins=20)
        cal_fig = display.figure_
        cal_fig.savefig(os.path.join(result_root_path, 'images', 'calibration_curve.jpg'))
        plt.close()
        plt.clf()
        
        plt.figure()
        bins = np.linspace(0, 1, 20)
        plt.hist(prob.numpy(), bins)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        
        save_path = os.path.join(result_root_path, 'images')
        os.makedirs(save_path, exist_ok=True)        
        
        plt.savefig(os.path.join(save_path, 'prob_histogram.jpg'))
        plt.close()
        plt.clf()


    def metric_ece(self, bin_size=0.1):
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1).numpy()
        target = target.cpu().numpy()
            
        prob = np.asarray(prob)
        target = np.asarray(target)
        
        total = len(prob)
        
        zero_class_conf = 1 - prob
        prob = np.stack((zero_class_conf, prob), axis=1)
        
        predictions = np.argmax(prob, axis=1)
        max_confs = np.amax(prob, axis=1)
        
        upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
        accs = []
        avg_confs = []
        bin_counts = []
        ces = []

        for upper_bound in upper_bounds:
            lower_bound = upper_bound - bin_size
            acc, avg_conf, bin_count = self.compute_bin(lower_bound, upper_bound, max_confs, predictions, target)
            accs.append(acc)
            avg_confs.append(avg_conf)
            bin_counts.append(bin_count)
            ces.append(abs(acc - avg_conf) * bin_count)

        ece = sum(ces) / total

        return ece
     
    def compute_bin(self, conf_thresh_lower, conf_thresh_upper, conf, pred, true):
        filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
        if len(filtered_tuples) < 1:
            return 0,0,0
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])
            avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
            accuracy = float(correct)/len(filtered_tuples)
            bin_count = len(filtered_tuples)
            return accuracy, avg_conf, bin_count
    


class MultiClassEvalMetrics(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
        **kwargs: Any,
    ) -> None:
        super(MultiClassEvalMetrics, self).__init__(**kwargs)
        self.multidim_average = multidim_average
        self._create_state(size=1, multidim_average=multidim_average)
        

    def _create_state(
        self,
        size: int,
        multidim_average: Literal["global", "samplewise"] = "samplewise",
    ) -> None:
        """Initialize the states for the different statistics."""
        default: Union[Callable[[], list], Callable[[], Tensor]]
        if multidim_average == "samplewise":
            default = list
            dist_reduce_fx = "cat"
        else:
            default = lambda: torch.zeros(size, dtype=torch.long)
            dist_reduce_fx = "sum"

        self.add_state("prob", default(), dist_reduce_fx=dist_reduce_fx)
        self.add_state("target", default(), dist_reduce_fx=dist_reduce_fx)
    
    def update(self, logit: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        prob = torch.softmax(logit.detach().float(), 1) # float16 -> float32 to prevent softmax failure 
        target = target.long().detach()
        
        if self.multidim_average == "samplewise":
            self.prob.append(prob)
            self.target.append(target)
    
    def _final_state(self) -> Tuple[Tensor, Tensor]:
        """Aggregate states that are lists and return final states."""
        prob = dim_zero_cat(self.prob)
        target = dim_zero_cat(self.target)
        
        return prob, target
    
    def compute(self) -> float:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        prob, target = self._final_state()
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        AUC = roc_auc_score(target, prob, multi_class='ovo')
        
        return AUC
        
    def on_epoch_end_compute(self, best_thres: float=None) -> Dict:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        
        metric_dict = {'Sensitivity': [], 'Specificity': [], 'Precision': [], 'Accuracy': [], 'F1_Score': []}
        
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
            
        # Compute AUC
        ovo_AUC = roc_auc_score(target, prob, multi_class='ovo')
        ovo_CI95_lower, ovo_CI95_upper = self.bootstrap_auc(prob, target, multi_class='ovo')
        
        AUC = roc_auc_score(target, prob, multi_class='ovr', average=None).tolist() # multiclass_auroc(prob, target, prob.shape[1], average='macro') # ovr
        CI95_lower, CI95_upper = self.bootstrap_auc(prob, target, multi_class='ovr')
        
        # Prediction
        pred = torch.argmax(prob, 1)
        
        acc=torch.sum(pred == target) / pred.shape[0]
        metric_dict['total_Accuracy']= float(acc)

        
        # Compute other metrics
        conf_mat = multiclass_confusion_matrix(pred, target, prob.shape[1]) # j axis : false negative, i axis : false positives
        
        
        metric_dict['Sensitivity']= []
        metric_dict['Specificity']=[]
        metric_dict['Precision']=[]
        metric_dict['Accuracy']=[]
        metric_dict['F1_Score']=[]
        metric_dict['CI95_lower']=[]
        metric_dict['CI95_upper']=[]
        
        for i in range(len(conf_mat)):        
            tp = conf_mat[i, i]
            fn = conf_mat[i].sum() - tp
            fp = conf_mat[:, i].sum() - tp
            tn = conf_mat.sum() - (tp + fn + fp)
            
            sensitivity = tp / max((tp + fn), 1e-6)
            specificity = tn / max((fp + tn), 1e-6)
            precision = tp / max((tp + fp), 1e-6)
            accuracy = (tp + tn) / max((tp + tn + fp + fn), 1e-6)
            f1 = 2 * precision * sensitivity / max((precision + sensitivity), 1e-6)
            
            metric_dict['Sensitivity'].append(float(sensitivity))
            metric_dict['Specificity'].append(float(specificity))
            metric_dict['Precision'].append(float(precision))
            metric_dict['Accuracy'].append(float(accuracy))
            metric_dict['F1_Score'].append(float(f1))
        
        metric_dict['AUC'] = AUC
        metric_dict['CI95_lower'] = CI95_lower
        metric_dict['CI95_upper'] = CI95_upper
        
        metric_dict['ovo_AUC'] = ovo_AUC
        metric_dict['ovo_CI95_lower'] = ovo_CI95_lower
        metric_dict['ovo_CI95_upper'] = ovo_CI95_upper
        
        return metric_dict
        
    def bootstrap_auc(self, prob, target, multi_class='ovr', n_bootstraps=5000, alpha=0.05) -> Tuple[float, float]:
        assert len(target) == len(prob)
        n = len(target)
        auc_scores = []

        # Bootstrap resampling
        for _ in range(n_bootstraps):
            indices = torch.randint(0, len(target), (n,))
            if multi_class == 'ovr':
                auc = roc_auc_score(target[indices], prob[indices], multi_class= multi_class, average=None)
            else:
                auc = roc_auc_score(target[indices], prob[indices], multi_class= multi_class)
            auc_scores.append(auc)

        # Compute Confidence Interval
        if multi_class == 'ovr':
            auc_scores = np.stack(auc_scores, 0)
            lower = torch.quantile(torch.tensor(auc_scores), (alpha / 2.), 0).tolist()
            upper = torch.quantile(torch.tensor(auc_scores), (1 - alpha / 2.), 0).tolist()
        else:
            sorted_scores = torch.tensor(sorted(auc_scores))
            lower = torch.quantile(sorted_scores, (alpha / 2.)).item()
            upper = torch.quantile(sorted_scores, (1 - alpha / 2.)).item()

        return lower, upper
    
    def plot_graphs(self, result_root_path):
        self.plot_roc_curve(result_root_path)
        
    def plot_roc_curve(self, result_root_path):
        prob, target = self._final_state()
        
        prob = prob.cpu().squeeze(1)
        target = target.cpu()
        fpr, tpr, _ = multiclass_roc(prob, target, prob.shape[1])
        
        for i in range(len(fpr)):            
            final_auc = auc(fpr[i].numpy(), tpr[i].numpy())
            plt.plot(fpr[i].numpy(), tpr[i].numpy(), label='%s ROC (area = %0.2f)' % (str(i), final_auc))
        
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        save_path = os.path.join(result_root_path, 'images')
        os.makedirs(save_path, exist_ok=True)
        
        plt.savefig(os.path.join(save_path, 'ROC_curve.jpg'))
        plt.close()
        plt.clf()
    

    
    


class MetricManager(object):
    def __init__(self, metric_fns):
        self.metric_fns = metric_fns
        self.result_dict = defaultdict(float)
        self.num_samples = 0 
    
    def __call__(self, prediction, ground_truth):
        self.num_samples += len(prediction)
        for metric_fn in self.metric_fns:
            for p, gt in zip(prediction, ground_truth):
                res = metric_fn(p, gt)
                dict_key = metric_fn.__name__
                self.result_dict[dict_key] += res
            
    def get_results(self):
        res_dict = {}
        for key, val in self.result_dict.items():
            res_dict[key] = val / self.num_samples
        return res_dict
    
    def reset(self):
        self.num_samples = 0
        self.result_dict = defaultdict(float)



def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()

  

if __name__ == "__main__":
    import torch
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score, roc_auc_score

    pred = np.array([[0.1, 0.8, 0.1],
                    [0.7, 0.2, 0.1],
                    [0.6, 0.3, 0.1],
                    [0.1, 0.3, 0.6]])
    # 0 vs rest : [0.1, 0.9]  [0.7, 0.3], [0.6, 0.4], [0.1. 0.9]
    # 1 vs rest : [0.8, 0.2]  [0.2, 0.8], [0.3, 0.7], [0.1. 0.9]
    pred = np.array([0.1, 0.2, 0.7, 0.8])


    label = np.array([0, 0, 1, 0])
    # label = np.array([2, 0, 1, 0])
    
    auc_score = roc_auc_score(label, pred, multi_class='ovr', average=None)
    
    
    if not isinstance(auc_score, np.ndarray):
        auc_score = [auc_score]
    else:
        auc_score = auc_score.tolist()
    
    print(auc_score)
    print(type(auc_score))
