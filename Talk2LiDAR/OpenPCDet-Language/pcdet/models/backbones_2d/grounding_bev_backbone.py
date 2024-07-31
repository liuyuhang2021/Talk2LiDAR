import numpy as np
import torch
import torch.nn as nn
import clip
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GroundingBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS  # [5, 5]
            layer_strides = self.model_cfg.LAYER_STRIDES  # [1, 2]
            num_filters = self.model_cfg.NUM_FILTERS  # [128, 256]
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS  # [256, 256]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES  # [1, 2]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)  # 2
        c_in_list = [input_channels, *num_filters[:-1]]  # [256, 128]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        hidden_channel = 256
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.mlp = nn.Sequential(
            nn.Linear(512, 128, dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128, dtype=torch.float32)
        )
        self.text_projection = nn.Parameter(
            torch.empty(
                size=(512, hidden_channel),
                device=self.device,
                dtype=torch.float32
            )
        )
        self._weight_initialization(hidden_channel)

    def _weight_initialization(self, hidden_channel):
        nn.init.normal_(self.text_projection, std=hidden_channel ** -0.5)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']  # B 256 180 180

        text_token = data_dict['text_token'].to(self.device).long()  # B 77
        with torch.no_grad():
            word_feature, sentence_feature = self.clip_model.encode_text(text_token)
        # [B, 77, 512], [B, 512]  float16

        word_feature = word_feature.to(torch.float32)  # [B, 77, 512]
        word_feature = nn.functional.normalize(word_feature, dim=2)
        word_feature = self.mlp(word_feature)  # B 77 128

        data_dict['word_feature'] = word_feature

        sentence_feature = sentence_feature.to(torch.float32)  # [B, 512]
        sentence_feature = sentence_feature @ self.text_projection  # B 256
        sentence_feature = sentence_feature.unsqueeze(2).unsqueeze(3)  # B 256 1 1

        # 逐元素相乘
        # x = spatial_features * sentence_feature

        # 逐元素相加
        # broadcast_sentence_feature = sentence_feature.expand(-1, -1, 180, 180)
        # x = spatial_features + broadcast_sentence_feature


        # concat 之后连接conv
        broadcast_sentence_feature = sentence_feature.expand(-1, -1, 180, 180)
        x = torch.cat([spatial_features, broadcast_sentence_feature], dim=1)
        x = self.shared_conv(x)

        # x = spatial_features
        ups = []
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class GruLayer(nn.Module):
    def __init__(self, use_bidir=False,
                 emb_size=100, hidden_size=256, num_layers=4,
                 out_dim=128):
        super().__init__()
        self.use_bidir = use_bidir
        self.num_bidir = 2 if self.use_bidir else 1
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        in_dim = hidden_size * 2 if use_bidir else hidden_size
        self.mlps = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_dim, out_channels=out_dim, kernel_size=1),
        )

    def forward(self, glove_feature, text_length):
        """
        encode the input descriptions
        """

        word_embs = glove_feature  # [B, 77, 50]
        max_des_len = word_embs.shape[1]  # 77
        lang_feat = pack_padded_sequence(word_embs, text_length.cpu(), batch_first=True, enforce_sorted=False)

        # encode description
        lang_feat, hidden_feat = self.gru(
            lang_feat)  # lang_feat:[B, 77, 256] hidden_feat:[4, B, 256]
        # [D, num_layer, B, hidden_size] choose last gru layer hidden [1, B, 256]
        hidden_feat = hidden_feat.view(self.num_bidir, self.num_layers, hidden_feat.shape[1], hidden_feat.shape[2])[:,
                      -1, :, :]
        hidden_feat = hidden_feat.permute(1, 0, 2).contiguous().flatten(start_dim=1)  # [B, 256]

        lang_feat, _ = pad_packed_sequence(lang_feat, batch_first=True, total_length=max_des_len)

        lang_feat = lang_feat.transpose(-1, -2)  # [B, C, N]
        lang_feat = self.mlps(lang_feat)
        lang_feat = lang_feat.transpose(-1, -2)  # [B, N, C]

        return lang_feat, hidden_feat

"""
tensor1 = torch.zeros(4, 77, 50)
tensor2 = torch.ones(4)

layer = GruLayer()
output_tensor1, output_tensor2 = layer(tensor1, tensor2)
# print(output_tensor1.shape)  [4, 77, 128]
# print(output_tensor2.shape)  [4, 256]
"""


class GroundingBEVBackbone_GRU(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS  # [5, 5]
            layer_strides = self.model_cfg.LAYER_STRIDES  # [1, 2]
            num_filters = self.model_cfg.NUM_FILTERS  # [128, 256]
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS  # [256, 256]
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES  # [1, 2]
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)  # 2
        c_in_list = [input_channels, *num_filters[:-1]]  # [256, 128]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

        self.GRU = GruLayer()


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']  # B 256 180 180

        glove_feature = data_dict['glove_feature'].to(self.device) # B 77 50
        text_length = data_dict["length"]  # B

        word_feature, sentence_feature = self.GRU(glove_feature, text_length)
        data_dict['word_feature'] = word_feature

        sentence_feature = sentence_feature.unsqueeze(2).unsqueeze(3)  # B 256 1 1

        # 逐元素相乘
        # x = spatial_features * sentence_feature

        # 逐元素相加
        # broadcast_sentence_feature = sentence_feature.expand(-1, -1, 180, 180)
        # x = spatial_features + broadcast_sentence_feature

        # concat 之后连接conv
        broadcast_sentence_feature = sentence_feature.expand(-1, -1, 180, 180)
        x = torch.cat([spatial_features, broadcast_sentence_feature], dim=1)
        x = self.shared_conv(x)

        ups = []
        ret_dict = {}
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict