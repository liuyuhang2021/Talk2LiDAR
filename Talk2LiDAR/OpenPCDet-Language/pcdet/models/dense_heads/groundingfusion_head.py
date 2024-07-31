import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils.transfusion_utils import clip_sigmoid
from ..model_utils.basic_block_2d import BasicBlock2D
from ..model_utils.transfusion_utils import PositionEmbeddingLearned, TransformerDecoderLayer
from .target_assigner.hungarian_assigner import HungarianAssigner3D
from ...utils import loss_utils
from ..model_utils import centernet_utils

import clip


class SeparateHead_Transfusion(nn.Module):
    def __init__(self, input_channels, head_channels, kernel_size, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv1d(input_channels, head_channels, kernel_size, stride=1, padding=kernel_size // 2,
                              bias=use_bias),
                    nn.BatchNorm1d(head_channels),
                    nn.ReLU()
                ))
            fc_list.append(
                nn.Conv1d(head_channels, output_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class GroundingFusionHead(nn.Module):
    """
        This module implements TransFusionHead.
        The code is adapted from https://github.com/mit-han-lab/bevfusion/ with minimal modifications.
    """

    def __init__(
            self,
            model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
            predict_boxes_when_training=True,
    ):
        super(GroundingFusionHead, self).__init__()

        self.grid_size = grid_size  #
        self.point_cloud_range = point_cloud_range  # [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
        self.voxel_size = voxel_size  # [0.075, 0.075, 0.2]
        self.num_classes = num_class  # 1

        self.model_cfg = model_cfg
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)  # 8
        self.dataset_name = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('DATASET', 'nuScenes')  # nuScenes

        hidden_channel = self.model_cfg.HIDDEN_CHANNEL  # 128
        self.num_proposals = self.model_cfg.NUM_PROPOSALS  # 200
        self.bn_momentum = self.model_cfg.BN_MOMENTUM  # 0.1
        self.nms_kernel_size = self.model_cfg.NMS_KERNEL_SIZE  # 3

        num_heads = self.model_cfg.NUM_HEADS  # 8
        dropout = self.model_cfg.DROPOUT  # 0.1
        activation = self.model_cfg.ACTIVATION  # relu
        ffn_channel = self.model_cfg.FFN_CHANNEL  # 256
        bias = self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)  # false

        loss_cls = self.model_cfg.LOSS_CONFIG.LOSS_CLS
        #             LOSS_CLS:
        #                 use_sigmoid: True
        #                 gamma: 2.0
        #                 alpha: 0.25
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)  # true
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = loss_utils.SigmoidFocalClassificationLoss(gamma=loss_cls.gamma, alpha=loss_cls.alpha)
        self.loss_cls_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']  # 1
        self.loss_bbox = loss_utils.L1Loss()
        self.loss_bbox_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['bbox_weight']  # 0.25
        self.loss_heatmap = loss_utils.GaussianFocalLoss()
        self.loss_heatmap_weight = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['hm_weight']  # 1

        self.code_size = 8

        # a shared convolution
        self.shared_conv = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channel, kernel_size=3, padding=1)

        layers = []
        layers.append(BasicBlock2D(hidden_channel, hidden_channel, kernel_size=3, padding=1, bias=bias))
        layers.append(nn.Conv2d(in_channels=hidden_channel, out_channels=1, kernel_size=3, padding=1))
        self.heatmap_head = nn.Sequential(*layers)

        self.class_encoding = nn.Conv1d(num_class, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                                               self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                                               cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                                               )  # 128 8 256 0.1

        self.word_decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                                                self_posembed=None, cross_posembed=None, cross_only=True)
        self.sentence_decoder = TransformerDecoderLayer(hidden_channel, num_heads, ffn_channel, dropout, activation,
                                                self_posembed=None, cross_posembed=None, cross_only=True)

        # Prediction Head
        heads = copy.deepcopy(self.model_cfg.SEPARATE_HEAD_CFG.HEAD_DICT)
        heads['heatmap'] = dict(out_channels=self.num_classes, num_conv=self.model_cfg.NUM_HM_CONV)  # 1 2
        self.prediction_head = SeparateHead_Transfusion(hidden_channel, 64, 1, heads, use_bias=bias)

        self.init_weights()
        self.bbox_assigner = HungarianAssigner3D(**self.model_cfg.TARGET_ASSIGNER_CONFIG.HUNGARIAN_ASSIGNER)

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.grid_size[0] // self.feature_map_stride  # 180
        y_size = self.grid_size[1] // self.feature_map_stride  # 180
        self.bev_pos = self.create_2D_grid(x_size, y_size)  # 180 180  BEV的位置embedding

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden_channel, dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel, dtype=torch.float32)
        )
        self.text_projection = nn.Parameter(
            torch.empty(
                size=(512, hidden_channel),
                device=self.device,
                dtype=torch.float32
            )
        )
        self._weight_initialization(hidden_channel)

        self.forward_ret_dict = {}

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.word_decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.sentence_decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, "query"):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _weight_initialization(self, hidden_channel):
        nn.init.normal_(self.text_projection, std=hidden_channel ** -0.5)

    def predict(self, inputs, word_feature, sentence_feature):
        batch_size = inputs.shape[0]  # B
        lidar_feat = self.shared_conv(inputs)  # B 128 180 180

        lidar_feat_flatten = lidar_feat.view(
            batch_size, lidar_feat.shape[1], -1
        )  # B 128 180*180
        word_feature = word_feature.permute(0, 2, 1).contiguous()  # B 128 77
        sentence_feature = sentence_feature.permute(0, 2, 1).contiguous()  # B 128 1

        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # B 180*180 2

        # query initialization
        dense_heatmap = self.heatmap_head(lidar_feat)  # B 1 180 180
        heatmap = dense_heatmap.detach().sigmoid()  # B 1 180 180，从计算图中分离
        padding = self.nms_kernel_size // 2  # 1
        local_max = torch.zeros_like(heatmap)  # B 1 180 180
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
        )  # B 1 178 178
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner  # B 1 180 180
        heatmap = heatmap * (heatmap == local_max)  # 非局部最大位置为0
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)  # B 1 180*180

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[
                        ..., : self.num_proposals
                        ]  # 200个proposal位置
        top_proposals_class = top_proposals // heatmap.shape[-1]  # 全为0  (B, 200)
        top_proposals_index = top_proposals % heatmap.shape[-1]  # 索引位置  (B, 200)
        query_feat = lidar_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1),
            dim=-1,
        )  # B 128 200
        self.query_labels = top_proposals_class  # B 200

        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]),
            dim=1,
        )  # B 200 2
        # convert to xy
        query_pos = query_pos.flip(dims=[-1])
        bev_pos = bev_pos.flip(dims=[-1])

        query_feat = self.decoder(
            query_feat, lidar_feat_flatten, query_pos, bev_pos
        )  # B 128 200, float 32
        query_feat = self.word_decoder(query_feat, word_feature, query_pos, bev_pos)
        query_feat = self.sentence_decoder(query_feat, sentence_feature, query_pos, bev_pos)  # (B, 128, 200)


        res_layer = self.prediction_head(query_feat)
        res_layer["center"] = res_layer["center"] + query_pos.permute(0, 2, 1)  # B 2 200

        res_layer["query_heatmap_score"] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),  # B 1 200
            dim=-1,
        )
        res_layer["dense_heatmap"] = dense_heatmap  # B 1 180 180

        return res_layer

    def forward(self, batch_dict):
        feats = batch_dict['spatial_features_2d']  # B 256 180 180
        text_token = batch_dict['text_token'].to(self.device).long()  # B 77

        with torch.no_grad():
            word_feature, sentence_feature = self.clip_model.encode_text(text_token)
        # [B, 77, 512], [B, 512]  float16

        word_feature = word_feature.to(torch.float32)  # [B, 77, 512]
        sentence_feature = sentence_feature.to(torch.float32)  # [B, 512]

        word_feature = nn.functional.normalize(word_feature, dim=2)


        word_feature = self.mlp(word_feature)  # B 77 128
        sentence_feature = sentence_feature @ self.text_projection  # B 128
        sentence_feature = sentence_feature.unsqueeze(1)  # B 1 128

        res = self.predict(feats, word_feature, sentence_feature)

        if not self.training:
            bboxes = self.get_bboxes(res)
            batch_dict['final_box_dicts'] = bboxes
        else:
            gt_boxes = batch_dict['gt_boxes']
            gt_bboxes_3d = gt_boxes[..., :-1]  # (B, 7)
            gt_labels_3d = gt_boxes[..., -1].long() - 1  # (B, 1) 全为0
            loss, tb_dict = self.loss(gt_bboxes_3d, gt_labels_3d, res)
            batch_dict['loss'] = loss
            batch_dict['tb_dict'] = tb_dict
        return batch_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, pred_dicts):
        assign_results = []
        for batch_idx in range(len(gt_bboxes_3d)):  # 遍历batch
            pred_dict = {}
            for key in pred_dicts.keys():
                pred_dict[key] = pred_dicts[key][batch_idx: batch_idx + 1]  # (1, *, 200)
            gt_bboxes = gt_bboxes_3d[batch_idx]  # (1, 7)
            valid_idx = []
            # filter empty boxes
            for i in range(len(gt_bboxes)):
                if gt_bboxes[i][3] > 0 and gt_bboxes[i][4] > 0:
                    valid_idx.append(i)
            assign_result = self.get_targets_single(gt_bboxes[valid_idx], gt_labels_3d[batch_idx][valid_idx], pred_dict)
            assign_results.append(assign_result)

        res_tuple = tuple(map(list, zip(*assign_results)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        num_pos = np.sum(res_tuple[4])
        matched_ious = np.mean(res_tuple[5])
        heatmap = torch.cat(res_tuple[6], dim=0)
        return labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        # gt_bboxes_3d: (1, 7)
        # gt_labels_3d: (1, 1)
        num_proposals = preds_dict["center"].shape[-1]  # 200
        score = copy.deepcopy(preds_dict["heatmap"].detach())  # 1 1 200
        center = copy.deepcopy(preds_dict["center"].detach())
        height = copy.deepcopy(preds_dict["height"].detach())
        dim = copy.deepcopy(preds_dict["dim"].detach())
        rot = copy.deepcopy(preds_dict["rot"].detach())  # 1 2 200
        if "vel" in preds_dict.keys():
            vel = copy.deepcopy(preds_dict["vel"].detach())
        else:
            vel = None

        boxes_dict = self.decode_bbox(score, rot, dim, center, height, vel)  # 从预测张量解耦为box
        bboxes_tensor = boxes_dict[0]["pred_boxes"]  # (200, 7)
        gt_bboxes_tensor = gt_bboxes_3d.to(score.device)  # (1, 7)

        assigned_gt_inds, ious = self.bbox_assigner.assign(
            bboxes_tensor, gt_bboxes_tensor, gt_labels_3d,
            score, self.point_cloud_range,
        )
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1  # gt 边界框索引
        if gt_bboxes_3d.numel() == 0:
            assert pos_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes_3d).view(-1, 7)
        else:
            pos_gt_bboxes = gt_bboxes_3d[pos_assigned_gt_inds.long(), :]

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.code_size]).to(center.device)  # (200, 8)
        bbox_weights = torch.zeros([num_proposals, self.code_size]).to(center.device)  # (200, 8)
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)  # (200)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)  # (200)

        '''
        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes
        '''

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.encode_bbox(pos_gt_bboxes)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = 1
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # compute dense heatmap targets
        device = labels.device
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        feature_map_size = (self.grid_size[:2] // self.feature_map_stride)
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])  # (1, 180, 180)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / self.voxel_size[0] / self.feature_map_stride
            length = length / self.voxel_size[1] / self.feature_map_stride
            if width > 0 and length > 0:
                radius = \
                centernet_utils.gaussian_radius(length.view(-1), width.view(-1), target_assigner_cfg.GAUSSIAN_OVERLAP)[
                    0]
                radius = max(target_assigner_cfg.MIN_RADIUS, int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.feature_map_stride
                coor_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.feature_map_stride

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                centernet_utils.draw_gaussian_to_heatmap(heatmap[gt_labels_3d[idx]], center_int, radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], int(pos_inds.shape[0]),
                float(mean_iou), heatmap[None])

    def loss(self, gt_bboxes_3d, gt_labels_3d, pred_dicts, **kwargs):

        labels, label_weights, bbox_targets, bbox_weights, num_pos, matched_ious, heatmap = \
            self.get_targets(gt_bboxes_3d, gt_labels_3d, pred_dicts)
        loss_dict = dict()
        loss_all = 0

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(pred_dicts["dense_heatmap"]),
            heatmap,
        ).sum() / max(heatmap.eq(1).float().sum().item(), 1)
        loss_dict["loss_heatmap"] = loss_heatmap.item() * self.loss_heatmap_weight
        loss_all += loss_heatmap * self.loss_heatmap_weight

        labels = labels.reshape(-1)  # (B*200)
        label_weights = label_weights.reshape(-1)  # (B*200)
        cls_score = pred_dicts["heatmap"].permute(0, 2, 1).reshape(-1, self.num_classes)  # # (B*200, 1)

        one_hot_targets = labels.unsqueeze(dim=-1)  # (B*200, 1) gt为1
        loss_cls = self.loss_cls(
            cls_score, one_hot_targets, label_weights
        ).sum() / max(num_pos, 1)

        preds = torch.cat([pred_dicts[head_name] for head_name in self.model_cfg.SEPARATE_HEAD_CFG.HEAD_ORDER],
                          dim=1).permute(0, 2, 1)
        code_weights = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights']
        reg_weights = bbox_weights * bbox_weights.new_tensor(code_weights)

        loss_bbox = self.loss_bbox(preds, bbox_targets)
        loss_bbox = (loss_bbox * reg_weights).sum() / max(num_pos, 1)

        loss_dict["loss_cls"] = loss_cls.item() * self.loss_cls_weight
        loss_dict["loss_bbox"] = loss_bbox.item() * self.loss_bbox_weight
        loss_all = loss_all + loss_cls * self.loss_cls_weight + loss_bbox * self.loss_bbox_weight

        loss_dict[f"matched_ious"] = loss_cls.new_tensor(matched_ious)
        loss_dict['loss_trans'] = loss_all

        return loss_all, loss_dict

    def encode_bbox(self, bboxes):
        targets = torch.zeros([bboxes.shape[0], 8]).to(bboxes.device)
        targets[:, 0] = (bboxes[:, 0] - self.point_cloud_range[0]) / (self.feature_map_stride * self.voxel_size[0])
        targets[:, 1] = (bboxes[:, 1] - self.point_cloud_range[1]) / (self.feature_map_stride * self.voxel_size[1])
        targets[:, 3:6] = bboxes[:, 3:6].log()
        targets[:, 2] = bboxes[:, 2]
        targets[:, 6] = torch.sin(bboxes[:, 6])
        targets[:, 7] = torch.cos(bboxes[:, 6])
        return targets

    def decode_bbox(self, heatmap, rot, dim, center, height, vel, filter=False):

        post_process_cfg = self.model_cfg.POST_PROCESSING
        score_thresh = post_process_cfg.SCORE_THRESH  # 0
        post_center_range = post_process_cfg.POST_CENTER_RANGE  # [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        post_center_range = torch.tensor(post_center_range).cuda().float()
        # class label
        final_preds = heatmap.max(1, keepdims=False).indices  # 类别 (B, 200) 全为0
        final_scores = heatmap.max(1, keepdims=False).values  # 置信度 (B, 200)

        center[:, 0, :] = center[:, 0, :] * self.feature_map_stride * self.voxel_size[0] + self.point_cloud_range[0]
        center[:, 1, :] = center[:, 1, :] * self.feature_map_stride * self.voxel_size[1] + self.point_cloud_range[1]
        dim = dim.exp()
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)  # (B, 200, 7)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            boxes3d = final_box_preds[i]
            scores = final_scores[i]
            labels = final_preds[i]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels
            }
            predictions_dicts.append(predictions_dict)

        if filter is False:
            return predictions_dicts

        thresh_mask = final_scores > score_thresh
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            cmask &= thresh_mask[i]

            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            predictions_dict = {
                'pred_boxes': boxes3d,
                'pred_scores': scores,
                'pred_labels': labels,
            }

            predictions_dicts.append(predictions_dict)

        return predictions_dicts

    def get_bboxes(self, preds_dicts):

        batch_size = preds_dicts["heatmap"].shape[0]  # (B, 1, 200)
        batch_score = preds_dicts["heatmap"].sigmoid()  # (B, 1, 200)
        batch_score = batch_score * preds_dicts["query_heatmap_score"]  # (B, 1, 200)
        batch_center = preds_dicts["center"]
        batch_height = preds_dicts["height"]
        batch_dim = preds_dicts["dim"]
        batch_rot = preds_dicts["rot"]
        batch_vel = None
        if "vel" in preds_dicts:
            batch_vel = preds_dicts["vel"]

        ret_dict = self.decode_bbox(
            batch_score, batch_rot, batch_dim,
            batch_center, batch_height, batch_vel,
            filter=True,
        )  # 长度为batch_size的list，每个元素为一个dict
        for k in range(batch_size):
            ret_dict[k]['pred_labels'] = ret_dict[k]['pred_labels'].int() + 1  # 全部为1

        return ret_dict
