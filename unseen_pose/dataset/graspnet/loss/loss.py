import torch
import torch.nn as nn
import numpy as np
from ..graspnet_constant import NUM_FUSION_FEATURE
import logging
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

def cosine(t1, t2):
    '''
    t1: (a, b)
    t2: (a, b) 
    return (a)
    '''
    product = t1 * t2 # (a, b)
    inner = torch.sum(product, dim=1) # (a, b)
    cosine_value = inner / torch.norm(t1, dim = 1) / torch.norm(t2, dim = 1) # (a)
    return cosine_value

class GraspNetLoss(nn.Module):
    def __init__(self, non_matches_thresh=0.4, matches_thresh=0.1, num_fusion_feature = NUM_FUSION_FEATURE, objectness_loss_weight=0.6):
        super().__init__()
        self.non_matches_thresh = non_matches_thresh
        self.matches_thresh = matches_thresh
        self.num_fusion_feature = num_fusion_feature
        self.objectness_loss_weight = objectness_loss_weight
        self.bceloss = nn.BCELoss()

        ce_weight_11 = torch.from_numpy(np.array([1, 2])).type(torch.float)
        self.celoss_11 = nn.CrossEntropyLoss(weight=ce_weight_11)

        ce_weight_12 = torch.from_numpy(np.array([1, 2])).type(torch.float)
        self.celoss_12 = nn.CrossEntropyLoss(weight=ce_weight_12)

    def forward(self, fusion_feature, objectness_prediction_result, verbose=False):
        '''
        **Input:**

        - fusion_feature: dict.
        '''
        # matches_feature_left = fusion_feature['l']['matches_feature'].reshape((-1, self.num_fusion_feature)) # (b * num_matches, 256)
        # matches_feature_right = fusion_feature['r']['matches_feature'].reshape((-1, self.num_fusion_feature)) # (b * num_non_matches, 256)
        # non_matches_feature_left = fusion_feature['l']['non_matches_feature'].reshape((-1, self.num_fusion_feature)) # (b * num_matches, 256)
        # non_matches_feature_right = fusion_feature['r']['non_matches_feature'].reshape((-1, self.num_fusion_feature)) # (b * num_non_matches, 256)
        # matches_angle = torch.acos(cosine(matches_feature_left, matches_feature_right)) # (b * num_matches)
        # non_matches_angle = torch.acos(cosine(non_matches_feature_left, non_matches_feature_right)) # (b * num_non_matches)
        # logger.debug(f'\033[034mmatches_feature:{matches_feature_left}')
        # # print(f'non_matches_angle:{non_matches_angle[:10]}')
        # # print(f'matches_angle:{matches_angle[:10]}')
        #
        # num_non_matches = len(non_matches_angle)
        # num_matches = len(matches_angle)
        #
        # if num_non_matches == 0:
        #     num_non_matches = 0.0001
        # if num_matches == 0:
        #     num_matches = 0.0001
        #
        # # print(f'num_non_matches:{num_non_matches}')
        # num_non_matches_above_thresh = max(torch.sum(non_matches_angle < self.non_matches_thresh), torch.tensor(0.0001, device = non_matches_angle.device))
        # # print(f'num_non_matches_above_thresh:{num_non_matches_above_thresh}')
        # non_match_above_thresh_ratio = num_non_matches_above_thresh / num_non_matches
        #
        # num_matches_above_thresh = max(torch.sum(matches_angle > self.matches_thresh), torch.tensor(0.0001, device=matches_angle.device))
        # match_above_thresh_ratio = num_matches_above_thresh / num_matches
        #
        # # matches_loss = torch.sum(matches_angle) / len(matches_angle)
        # # print(f'self.non_matches_thresh - non_matches_angle:{self.non_matches_thresh - non_matches_angle}\033[0m')
        # non_matches_loss = torch.sum(torch.max(self.non_matches_thresh - non_matches_angle, torch.zeros((num_non_matches), dtype=float, device = non_matches_angle.device))) / num_non_matches_above_thresh
        # matches_loss = torch.sum(torch.max(matches_angle - self.matches_thresh, torch.zeros((num_matches), dtype=float, device = non_matches_angle.device))) / num_matches_above_thresh

        # Objectness loss
        objectness_prediction =  objectness_prediction_result['sem_seg_pred'].reshape((-1, 2))
        objectness_label = objectness_prediction_result['sem_seg_label'].reshape((-1, )).type(torch.int64)


        # Match loss
        match_prediction = fusion_feature['match_pred'].reshape((-1, 2))
        match_label = fusion_feature['match_label'].reshape((-1, )).type(torch.int64)

        # FOR DEBUG
        # print(objectness_label[:20])
        # print(objectness_prediction[:20])

        res = (torch.nn.functional.softmax(objectness_prediction, dim=1)[:, 1] > 0.5).type(torch.int64)
        # print(objectness_prediction[:, 1])
        objectness_loss = self.celoss_12(objectness_prediction, objectness_label.type(torch.int64))
        objectness_f1 = f1_score(objectness_label.cpu().numpy(), res.cpu().numpy(), average='binary')
        objectness_confusion_matrix = confusion_matrix(objectness_label.cpu().numpy(), res.cpu().numpy())

        # print("binary classification loss for semantic seg: ", objectness_loss)
        if verbose:
            print("confusion matrix for semantic seg: ", objectness_confusion_matrix)

        res_match = (torch.nn.functional.softmax(match_prediction, dim=1)[:, 1] > 0.5).type(torch.int64)
        # print(objectness_prediction[:, 1])
        match_loss = self.celoss_11(match_prediction, match_label.type(torch.int64))
        match_f1 = f1_score(match_label.cpu().numpy(), res_match.cpu().numpy(), average='binary')
        match_confusion_matrix = confusion_matrix(match_label.cpu().numpy(), res_match.cpu().numpy())
        if verbose:
            print("confusion matrix for match prediction: ", match_confusion_matrix)

        loss = (1-self.objectness_loss_weight) * match_loss + objectness_loss * self.objectness_loss_weight
        # loss = (1-self.objectness_loss_weight) * (matches_loss + non_matches_loss) + objectness_loss * self.objectness_loss_weight
        # return loss, matches_loss, match_above_thresh_ratio, non_matches_loss, non_match_above_thresh_ratio, objectness_f1
        return loss, match_loss, objectness_loss, match_f1, objectness_f1
