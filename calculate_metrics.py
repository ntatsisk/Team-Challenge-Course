import copy
import numpy as np

def dice_score(ground_truth, prediction):
    """ Calculate dice score"""

    # Normalize
    prediction /= np.amax(prediction)
    ground_truth /= np.amax(ground_truth)

    true_positive_mask = np.logical_and(ground_truth==1, prediction==1)
    false_positive_mask = np.logical_and(ground_truth==0, prediction==1)
    false_negative_mask = np.logical_and(ground_truth==1, prediction==0)

    TP = np.count_nonzero(true_positive_mask)
    FP = np.count_nonzero(false_positive_mask)
    FN = np.count_nonzero(false_negative_mask)

    DSC = 2*TP / (2*TP + FP + FN)

    return DSC

def optimal_treshold(ground_truth, prediction, low=0.99, high=0.999999, steps=100):
    """ Find optimal probability threshold that maximizes dice score for a given
    segmentation prediction"""

    # Range of threshold values to be tested
    limits = np.linspace(low, high, steps)

    dice_scores = []
    for limit in limits:
        thres_seg = copy.deepcopy(prediction)
        thres_seg[thres_seg>limit] = 1
        thres_seg[thres_seg<=limit] = 0
        d = dice_score(ground_truth, thres_seg)
        dice_scores.append(d)

    # Find the optimal threshold (maximum dice score)
    opt_pos = np.argmax(dice_scores)
    opt_threshold = limits[opt_pos]
    opt_seg = copy.deepcopy(prediction)
    opt_seg[opt_seg>opt_threshold] = 1
    opt_seg[opt_seg<=opt_threshold] = 0
    opt_dice = dice_scores[opt_pos]

    return opt_seg, opt_threshold, opt_dice

def calculate_EF(ED_array, ES_array, spacing):
    """ """
    voxelvolume = spacing[0] * spacing[1] * spacing[2]

    ED_volume = np.sum(ED_array==1)*voxelvolume
    ES_volume = np.sum(ES_array==1)*voxelvolume

    strokevolume = ED_volume - ES_volume
    LV_EF = (strokevolume/ED_volume)*100

    return LV_EF, strokevolume
