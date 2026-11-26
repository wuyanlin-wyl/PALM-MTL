"""补充内容见evaluate.ipynb"""
from __future__ import division

import numpy as np
import six
import torch
import torch.nn as nn


def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.

    """
    pred_labels = iter(pred_labels)  # (352, 480)
    gt_labels = iter(gt_labels)  # (352, 480)

    n_class = 1
    confusion = np.zeros((n_class, n_class), dtype=np.int64)  # (12, 12)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        # if pred_label.ndim != 2 or gt_label.ndim != 2:
        #     raise ValueError('ndim of labels should be two.')
        # if pred_label.shape != gt_label.shape:
        #     raise ValueError('Shape of ground truth and prediction should'
        #                      ' be same.')
        pred_label = pred_label.flatten()  # (168960, )
        gt_label = gt_label.flatten()  # (168960, )

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        # print(lb_max)
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.  极度巧妙 × class_nums 正好使得每个ij能够对应.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) + pred_label[mask],
            minlength=n_class ** 2) \
            .reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')

    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.

    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.

    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.

    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
                       - np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou[:-1]
    # return iou


def eval_semantic_segmentation(pred_labels, gt_labels, preout, gtout):
    """Evaluate metrics used in Semantic Segmentation.

    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.

    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`

    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
            For example, this is a list of labels
            :obj:`[label_0, label_1, ...]`, where
            :obj:`label_i.shape = (H_i, W_i)`.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.

    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    """

    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / (np.sum(confusion, axis=1) + 1e-10)
    JS = get_JS(preout, gtout)
    DC = get_dice(preout, gtout)
    RVD = rvd(preout, gtout)
    VOE = _VOE(preout, gtout)
    SP = _specificity(preout, gtout)
    SE = _sensitivity(preout, gtout)
    PC = _precision(preout, gtout)
    RE = _recall(preout, gtout)
    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy[:-1]),
            'JS': JS,
            'DC': DC,
            'SP': SP,
            'SE': SE,
            'PC': PC,
            'RE': RE,
            'RVD': RVD,
            'VOE': VOE
            }
    # 'mean_class_accuracy': np.nanmean(class_accuracy)}


# JC/VOE
def get_JS(preout, gtout):
    # JS : Jaccard similarity

    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    intersection = torch.sum((preout + gtout) == 0)
    union = torch.sum((preout + gtout) == 0) + torch.sum((preout + gtout) == 1)

    JS = float(intersection) / (float(union) + 1e-6)

    return JS


# def get_jaccard(preout,gtout):
#     """
#     :return: 杰卡德系数
#     """
#     intersection = (gtout * preout).sum()
#     union = (gtout | preout).sum()
#
#     return intersection / union

# def dice(preout,gtout):
#     #     """
#     #     :return: dice系数 dice系数的分子 dice系数的分母(后两者用于计算dice_global)
#     #     """
#     #     # print('preout:',preout)
#     # print('gtout:',gtout)
#     intersection = (gtout * preout).sum()
#     union = gtout.sum() + preout.sum()
#
#     # print(intersection)
#     # print(union)
#     # print(2 * intersection / union)
#     return 2 * intersection / union

def get_dice(preout, gtout):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
    preout = torch.Tensor(preout)
    # print('preout_shape:',preout.shape)
    gtout = torch.Tensor(gtout)
    # print('gtout_shape:',gtout.shape)

    intersection = torch.sum((preout + gtout) == 0)
    # print('intersection',intersection)
    union = torch.sum((preout + gtout) == 0) + torch.sum((preout + gtout) == 0) + torch.sum((preout + gtout) == 1)
    # union1 = torch.sum((preout + gtout) != 2)
    # print('union',union)

    # print(intersection)
    # print(union)
    # print('DICE;',float(2 * intersection) / float(union))
    return float(2 * intersection) / float(union + 1e-6)


def rvd(preout, gtout):
    """
    :return: 体素相对误差 Relative Volume Difference
    """

    preout = torch.Tensor(preout)
    # print('preout_shape:',preout.shape)
    gtout = torch.Tensor(gtout)
    # print('gtout_shape:',gtout.shape)

    a = torch.sum(preout == 0)
    # print('A',a)
    b = torch.sum(gtout == 0)
    # print('B',b)

    return float(a - b) / float(b + 1e-6)


def _VOE(preout, gtout):
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    a = torch.sum(preout == 0)
    b = torch.sum(gtout == 0)

    VOE = 2 * (a - b) / (a + b)
    return VOE


def _specificity(preout, gtout):
    """特异性Specificity = TN / （FP + TN）"""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TN = ((preout == 1) & (gtout == 1))
    FP = ((preout == 0) & (gtout == 1))

    return float(torch.sum(TN)) / float(torch.sum(TN + FP) + 1e-6)


def _sensitivity(preout, gtout):
    """敏感性Sensitivity = TP / （TP + FN）"""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FN = ((preout == 1) & (gtout == 0))

    return float(torch.sum(TP)) / float(torch.sum(TP + FN) + 1e-6)


def _precision(preout, gtout):
    """查准率Precision = TP / （TP + FP）"""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FP = ((preout == 0) & (gtout == 1))

    return float(torch.sum(TP)) / float(torch.sum(TP + FP) + 1e-6)


def _recall(preout, gtout):
    """查全率Recall = TP / （TP + FN）"""
    preout = torch.Tensor(preout)
    gtout = torch.Tensor(gtout)

    TP = ((preout == 0) & (gtout == 0))
    FN = ((preout == 1) & (gtout == 0))

    return float(torch.sum(TP)) / float(torch.sum(TP + FN) + 1e-6)


# from scipy.ndimage import morphology
# from mindspore._checkparam import Validator as validator
# from .metric import Metric
#
#
# class MeanSurfaceDistance(Metric):
#
#     def __init__(self, symmetric=False, distance_metric="euclidean"):
#         super(MeanSurfaceDistance, self).__init__()
#         self.distance_metric_list = ["euclidean", "chessboard", "taxicab"]
#         distance_metric = validator.check_value_type("distance_metric", distance_metric, [str])
#         self.distance_metric = validator.check_string(distance_metric, self.distance_metric_list, "distance_metric")
#         self.symmetric = validator.check_value_type("symmetric", symmetric, [bool])
#         self.clear()
#
#     def clear(self):
#         """Clears the internal evaluation result."""
#         self._y_pred_edges = 0
#         self._y_edges = 0
#         self._is_update = False
#
#     def _get_surface_distance(self, y_pred_edges, y_edges):
#         """
#         计算从预测图片边界到真实图片边界的表面距离。
#         """
#
#         if not np.any(y_pred_edges):
#             return np.array([])
#
#         if not np.any(y_edges):
#             dis = np.full(y_edges.shape, np.inf)
#         else:
#             if self.distance_metric == "euclidean":
#                 dis = morphology.distance_transform_edt(~y_edges)
#             elif self.distance_metric in self.distance_metric_list[-2:]:
#                 dis = morphology.distance_transform_cdt(~y_edges, metric=self.distance_metric)
#
#         surface_distance = dis[y_pred_edges]
#
#         return surface_distance
#
#     def update(self, *inputs):
#         """
#         更新输入数据。
#         """
#         if len(inputs) != 3:
#             raise ValueError('MeanSurfaceDistance need 3 inputs (y_pred, y, label), but got {}.'.format(len(inputs)))
#         y_pred = self._convert_data(inputs[0])
#         y = self._convert_data(inputs[1])
#         label_idx = inputs[2]
#
#         if y_pred.size == 0 or y_pred.shape != y.shape:
#             raise ValueError("y_pred and y should have same shape, but got {}, {}.".format(y_pred.shape, y.shape))
#
#         if y_pred.dtype != bool:
#             y_pred = y_pred == label_idx
#         if y.dtype != bool:
#             y = y == label_idx
#
#         self._y_pred_edges = morphology.binary_erosion(y_pred) ^ y_pred
#         self._y_edges = morphology.binary_erosion(y) ^ y
#         self._is_update = True
#
#     def eval(self):
#         """
#         计算平均表面距离。
#         """
#         if self._is_update is False:
#             raise RuntimeError('Call the update method before calling eval.')
#
#         mean_surface_distance = self._get_surface_distance(self._y_pred_edges, self._y_edges)
#
#         if mean_surface_distance.shape == (0,):
#             return np.inf
#
#         avg_surface_distance = mean_surface_distance.mean()
#
#         if not self.symmetric:
#             return avg_surface_distance
#
#         contrary_mean_surface_distance = self._get_surface_distance(self._y_edges, self._y_pred_edges)
#         if contrary_mean_surface_distance.shape == (0,):
#             return np.inf
#
#         contrary_avg_surface_distance = contrary_mean_surface_distance.mean()
#         return np.mean((avg_surface_distance, contrary_avg_surface_distance))


if __name__ == "__main__":
    import torch as t
    import numpy

    print('-----' * 5)
    pred_labels = numpy.random.randint(6, 256, 256)
    gt_labels = numpy.random.randint(6, 256, 256)
    preout = numpy.random.randint(6, 256, 256)
    gtout = numpy.random.randint(6, 256, 256)
