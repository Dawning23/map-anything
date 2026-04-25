# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
MapAnything model class defined using UniCeption modules.
"""

import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from mapanything.utils.geometry import (
    apply_log_to_norm,
    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap,
    normalize_depth_using_non_zero_pixels,
    normalize_pose_translations,
    transform_pose_using_quats_and_trans_2_to_1,
)
from mapanything.utils.inference import (
    postprocess_model_outputs_for_inference,
    preprocess_input_views_for_inference,
    validate_input_views_for_inference,
)
from uniception.models.encoders import (
    encoder_factory,
    EncoderGlobalRepInput,
    ViTEncoderInput,
    ViTEncoderNonImageInput,
)
from uniception.models.info_sharing.alternating_attention_transformer import (
    MultiViewAlternatingAttentionTransformer,
    MultiViewAlternatingAttentionTransformerIFR,
)
from uniception.models.info_sharing.base import MultiViewTransformerInput
from uniception.models.info_sharing.cross_attention_transformer import (
    MultiViewCrossAttentionTransformer,
    MultiViewCrossAttentionTransformerIFR,
)
from uniception.models.info_sharing.global_attention_transformer import (
    MultiViewGlobalAttentionTransformer,
    MultiViewGlobalAttentionTransformerIFR,
)
from uniception.models.prediction_heads.adaptors import (
    CamTranslationPlusQuatsAdaptor,
    PointMapAdaptor,
    PointMapPlusRayDirectionsPlusDepthAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor,
    PointMapPlusRayDirectionsPlusDepthWithMaskAdaptor,
    PointMapWithConfidenceAdaptor,
    PointMapWithConfidenceAndMaskAdaptor,
    PointMapWithMaskAdaptor,
    RayDirectionsPlusDepthAdaptor,
    RayDirectionsPlusDepthWithConfidenceAdaptor,
    RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor,
    RayDirectionsPlusDepthWithMaskAdaptor,
    RayMapPlusDepthAdaptor,
    RayMapPlusDepthWithConfidenceAdaptor,
    RayMapPlusDepthWithConfidenceAndMaskAdaptor,
    RayMapPlusDepthWithMaskAdaptor,
    ScaleAdaptor,
)
from uniception.models.prediction_heads.base import (
    AdaptorInput,
    PredictionHeadInput,
    PredictionHeadLayeredInput,
    PredictionHeadTokenInput,
)
from uniception.models.prediction_heads.dpt import DPTFeature, DPTRegressionProcessor
from uniception.models.prediction_heads.linear import LinearFeature
from uniception.models.prediction_heads.mlp_head import MLPHead
from uniception.models.prediction_heads.pose_head import PoseHead
from uniception.models.utils.transformer_blocks import Mlp, SwiGLUFFNFused

# Enable TF32 precision if supported (for GPU >= Ampere and PyTorch >= 1.12)
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


# ========================================================================================
# 【核心类】MapAnything: 多视图3D场景重建模型
# 该模型是整个系统的核心，它将图像编码器、几何输入编码器、多视图Transformer和预测头
# 组合在一起，实现从多视图图像到3D场景表示（点云/深度/位姿）的端到端推理。
# 整体架构: 图像编码 -> 几何特征融合 -> 多视图信息共享 -> 预测头 -> 场景表示
# ========================================================================================
class MapAnything(nn.Module, PyTorchModelHubMixin):
    "模块化的MapAnything模型类，支持图像和可选几何模态的输入（多种重建任务）。"

    def __init__(
        self,
        name: str,  # 模型名称
        encoder_config: Dict,  # 图像编码器配置
        info_sharing_config: Dict,  # 多视图注意力Transformer配置
        pred_head_config: Dict,  # 预测头配置
        geometric_input_config: Dict,  # 几何输入模态配置（光线方向/深度/相机位姿）
        fusion_norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),  # 特征融合后的归一化层，默认使用LayerNorm
        pretrained_checkpoint_path: str = None,  # 预训练权重路径
        load_specific_pretrained_submodules: bool = False,  # 是否只加载特定子模块的预训练权重
        specific_pretrained_submodules: list = None,  # 要加载的特定子模块名称列表
        torch_hub_force_reload: bool = False,  # 是否强制从torch hub重新下载编码器
        use_register_tokens_from_encoder: bool = False,  # 是否使用编码器的register token
        info_sharing_mlp_layer_str: str = "mlp",  # 多视图Transformer中MLP层的类型
    ):
        """
        多视图模型：包含图像编码器、可选几何模态融合、多视图注意力Transformer和下游预测头。
        目标是输出场景的3D表示。
        多视图注意力Transformer还接收一个scale token，用于预测场景表示的度量缩放因子。
        """
        super().__init__()  # 调用父类nn.Module的初始化

        # ============================================================
        # 【重要代码段】保存所有初始化参数为类属性
        # 重要性：这些属性在后续的模型构建、推理配置和模型序列化中被反复使用。
        # class_init_args字典用于支持HuggingFace Hub的模型序列化/反序列化。
        # ============================================================
        self.name = name  # 存储模型名称
        self.encoder_config = encoder_config  # 存储编码器配置
        self.info_sharing_config = info_sharing_config  # 存储信息共享模块配置
        self.pred_head_config = pred_head_config  # 存储预测头配置
        self.geometric_input_config = geometric_input_config  # 存储几何输入配置
        self.pretrained_checkpoint_path = pretrained_checkpoint_path  # 存储预训练权重路径
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules  # 存储是否部分加载标志
        self.specific_pretrained_submodules = specific_pretrained_submodules  # 存储需要加载的子模块列表
        self.torch_hub_force_reload = torch_hub_force_reload  # 存储是否强制重载标志
        self.use_register_tokens_from_encoder = use_register_tokens_from_encoder  # 存储是否使用register token标志
        self.info_sharing_mlp_layer_str = info_sharing_mlp_layer_str  # 存储MLP类型字符串
        self.class_init_args = {  # 收集所有构造参数，用于模型的序列化（保存/加载）
            "name": self.name,
            "encoder_config": self.encoder_config,
            "info_sharing_config": self.info_sharing_config,
            "pred_head_config": self.pred_head_config,
            "geometric_input_config": self.geometric_input_config,
            "pretrained_checkpoint_path": self.pretrained_checkpoint_path,
            "load_specific_pretrained_submodules": self.load_specific_pretrained_submodules,
            "specific_pretrained_submodules": self.specific_pretrained_submodules,
            "torch_hub_force_reload": self.torch_hub_force_reload,
            "use_register_tokens_from_encoder": self.use_register_tokens_from_encoder,
            "info_sharing_mlp_layer_str": self.info_sharing_mlp_layer_str,
        }

        # 从配置中提取关键参数
        self.info_sharing_type = info_sharing_config["model_type"]  # 多视图Transformer类型（cross_attention/global_attention/alternating_attention）
        self.info_sharing_return_type = info_sharing_config["model_return_type"]  # 返回类型：是否包含中间层特征
        self.pred_head_type = pred_head_config["type"]  # 预测头类型（linear/dpt/dpt+pose）

        # ============================================================
        # 【重要代码段】初始化图像编码器（Image Encoder）
        # 重要性：图像编码器（如DINOv2 ViT）是整个模型的骨干网络，负责将输入图像
        # 编码为高维特征表示。它的输出是后续所有处理（几何融合、多视图注意力等）的基础。
        # ============================================================
        if self.encoder_config["uses_torch_hub"]:  # 如果编码器来自torch hub
            self.encoder_config["torch_hub_force_reload"] = torch_hub_force_reload  # 设置是否强制重新下载
        encoder_config_copy = self.encoder_config.copy()  # 复制配置（避免修改原始配置影响序列化）
        del encoder_config_copy["uses_torch_hub"]  # 删除工厂方法不需要的参数
        self.encoder = encoder_factory(**encoder_config_copy)  # 通过工厂方法创建图像编码器实例

        # ============================================================
        # 【重要代码段】初始化几何模态编码器集合
        # 重要性：这些编码器将不同的几何先验信息（光线方向、深度、相机旋转、相机平移及其尺度因子）
        # 编码为与图像特征相同维度的向量，使得后续可以通过逐元素相加进行特征融合。
        # 这是MapAnything支持可选几何输入的关键机制——几何信息作为条件注入到图像特征中。
        # ============================================================

        # 初始化光线方向编码器：将每像素的3D光线方向编码为特征
        ray_dirs_encoder_config = self.geometric_input_config["ray_dirs_encoder_config"]
        ray_dirs_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim  # 匹配图像编码器的特征维度
        ray_dirs_encoder_config["patch_size"] = self.encoder.patch_size  # 匹配图像编码器的patch大小
        self.ray_dirs_encoder = encoder_factory(**ray_dirs_encoder_config)  # 创建光线方向编码器

        # 初始化深度编码器：将归一化后的深度值（取对数后）编码为特征
        depth_encoder_config = self.geometric_input_config["depth_encoder_config"]
        depth_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim  # 匹配图像编码器的特征维度
        depth_encoder_config["patch_size"] = self.encoder.patch_size  # 匹配图像编码器的patch大小
        self.depth_encoder = encoder_factory(**depth_encoder_config)  # 创建深度编码器

        # 初始化深度尺度因子编码器：将深度的归一化因子（取对数后）编码为全局特征
        depth_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        depth_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim  # 匹配图像编码器的特征维度
        self.depth_scale_encoder = encoder_factory(**depth_scale_encoder_config)  # 创建深度尺度编码器

        # 初始化相机旋转编码器：将四元数形式的相机旋转编码为全局特征
        cam_rot_encoder_config = self.geometric_input_config["cam_rot_encoder_config"]
        cam_rot_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim  # 匹配图像编码器的特征维度
        self.cam_rot_encoder = encoder_factory(**cam_rot_encoder_config)  # 创建相机旋转编码器

        # 初始化相机平移编码器：将归一化后的相机平移编码为全局特征
        cam_trans_encoder_config = self.geometric_input_config[
            "cam_trans_encoder_config"
        ]
        cam_trans_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim  # 匹配图像编码器的特征维度
        self.cam_trans_encoder = encoder_factory(**cam_trans_encoder_config)  # 创建相机平移编码器

        # 初始化相机平移尺度因子编码器：将平移的归一化因子（取对数后）编码为全局特征
        cam_trans_scale_encoder_config = self.geometric_input_config[
            "scale_encoder_config"
        ]
        cam_trans_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim  # 匹配图像编码器的特征维度
        self.cam_trans_scale_encoder = encoder_factory(**cam_trans_scale_encoder_config)  # 创建相机平移尺度编码器

        # 初始化融合归一化层：在图像特征与几何特征相加融合后进行LayerNorm归一化
        self.fusion_norm_layer = fusion_norm_layer(self.encoder.enc_embed_dim)

        # ============================================================
        # 【重要代码段】初始化Scale Token（尺度令牌）
        # 重要性：Scale Token是一个可学习的参数向量，在多视图Transformer中与图像patch token
        # 一起参与注意力计算。它的作用是聚合全局信息来预测场景的度量缩放因子，
        # 使模型能够从相对尺度恢复到绝对度量尺度（以米为单位）。
        # ============================================================
        self.scale_token = nn.Parameter(torch.zeros(self.encoder.enc_embed_dim))  # 创建尺度令牌参数，维度与编码器嵌入维度相同
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)  # 使用截断正态分布初始化（std=0.02是ViT中常用的初始化标准差）

        # 为信息共享Transformer设置MLP层类型
        if info_sharing_mlp_layer_str == "mlp":
            info_sharing_config["module_args"]["mlp_layer"] = Mlp  # 使用标准MLP
        elif info_sharing_mlp_layer_str == "swiglufused":
            info_sharing_config["module_args"]["mlp_layer"] = SwiGLUFFNFused  # 使用SwiGLU激活的FFN（性能更好但计算量稍大）
        else:
            raise ValueError(
                f"Invalid info_sharing_mlp_layer_str: {info_sharing_mlp_layer_str}. Valid options: ['mlp', 'swiglufused']"
            )

        # 初始化信息共享模块（多视图Transformer）——跨视图特征交互的核心
        self._initialize_info_sharing(info_sharing_config)

        # 初始化预测头——将Transformer特征解码为具体的场景表示
        self._initialize_prediction_heads(pred_head_config)

        # 初始化适配器——将预测头的原始输出转换为最终的场景表示格式
        self._initialize_adaptors(pred_head_config)

        # 加载预训练权重（如果提供了路径）
        self._load_pretrained_weights()

    @property
    def device(self) -> torch.device:
        """获取模型参数所在的设备（CPU/GPU）"""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """获取模型参数的数据类型（float32/float16/bfloat16）"""
        return next(self.parameters()).dtype

    # ============================================================
    # 【重要代码段】初始化信息共享模块（多视图Transformer）
    # 重要性：这是MapAnything实现跨视图信息交互的核心模块。通过多视图注意力机制，
    # 不同视图的特征可以互相参考和融合，从而实现一致的3D场景理解。
    # 支持三种注意力类型：交叉注意力、全局注意力、交替注意力。
    # ============================================================
    def _initialize_info_sharing(self, info_sharing_config):
        """
        根据配置初始化信息共享模块。
        设置自定义位置编码（如果指定），并根据配置类型初始化对应的多视图Transformer。
        """
        # 初始化自定义位置编码（如需要）
        custom_positional_encoding = info_sharing_config["custom_positional_encoding"]
        if custom_positional_encoding is not None:
            if isinstance(custom_positional_encoding, str):  # 字符串形式的位置编码（暂未实现）
                print(
                    f"Using custom positional encoding for multi-view attention transformer: {custom_positional_encoding}"
                )
                raise ValueError(
                    f"Invalid custom_positional_encoding: {custom_positional_encoding}. None implemented."
                )
            elif isinstance(custom_positional_encoding, Callable):  # 可调用函数形式的位置编码
                print(
                    "Using callable function as custom positional encoding for multi-view attention transformer."
                )
                self.custom_positional_encoding = custom_positional_encoding
        else:
            self.custom_positional_encoding = None  # 不使用自定义位置编码

        # 将图像编码器的嵌入维度和自定义位置编码注入到信息共享配置中
        info_sharing_config["module_args"]["input_embed_dim"] = (
            self.encoder.enc_embed_dim  # 确保Transformer的输入维度与编码器输出维度匹配
        )
        info_sharing_config["module_args"]["custom_positional_encoding"] = (
            self.custom_positional_encoding
        )

        # 根据返回类型和注意力类型初始化多视图Transformer
        if self.info_sharing_return_type == "no_intermediate_features":
            # 模式一：仅返回最后一层归一化后的特征（用于linear预测头）
            if self.info_sharing_type == "cross_attention":
                self.info_sharing = MultiViewCrossAttentionTransformer(  # 交叉注意力：视图间两两交互
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "global_attention":
                self.info_sharing = MultiViewGlobalAttentionTransformer(  # 全局注意力：所有视图拼接后做自注意力
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "alternating_attention":
                self.info_sharing = MultiViewAlternatingAttentionTransformer(  # 交替注意力：交替执行视图内和视图间注意力
                    **info_sharing_config["module_args"]
                )
            else:
                raise ValueError(
                    f"Invalid info_sharing_type: {self.info_sharing_type}. Valid options: ['cross_attention', 'global_attention', 'alternating_attention']"
                )
        elif self.info_sharing_return_type == "intermediate_features":
            # 模式二：返回中间层特征和最后一层特征（用于DPT预测头，DPT需要多尺度特征）
            if self.info_sharing_type == "cross_attention":
                self.info_sharing = MultiViewCrossAttentionTransformerIFR(  # IFR = Intermediate Feature Returner
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "global_attention":
                self.info_sharing = MultiViewGlobalAttentionTransformerIFR(
                    **info_sharing_config["module_args"]
                )
            elif self.info_sharing_type == "alternating_attention":
                self.info_sharing = MultiViewAlternatingAttentionTransformerIFR(
                    **info_sharing_config["module_args"]
                )
            else:
                raise ValueError(
                    f"Invalid info_sharing_type: {self.info_sharing_type}. Valid options: ['cross_attention', 'global_attention', 'alternating_attention']"
                )
            # 判断DPT是否需要使用编码器原始特征作为额外输入
            # DPT需要4层特征输入：如果Transformer只返回2层中间特征，则需补充1层编码器特征
            if len(self.info_sharing.indices) == 2:
                self.use_encoder_features_for_dpt = True  # Transformer返回2层中间特征，需要编码器特征补充（共4层）
            elif len(self.info_sharing.indices) == 3:
                self.use_encoder_features_for_dpt = False  # Transformer返回3层中间特征，加上最终层共4层，无需编码器特征
            else:
                raise ValueError(
                    "Invalid number of indices provided for info sharing feature returner. Please provide 2 or 3 indices."
                )
        else:
            raise ValueError(
                f"Invalid info_sharing_return_type: {self.info_sharing_return_type}. Valid options: ['no_intermediate_features', 'intermediate_features']"
            )

    # ============================================================
    # 【重要代码段】初始化预测头
    # 重要性：预测头负责将Transformer的特征解码为具体的3D场景表示（点云/深度/位姿等）。
    # 支持三种模式：linear（简单线性解码）、dpt（Dense Prediction Transformer，多尺度解码）、
    # dpt+pose（在DPT基础上增加相机位姿预测头）。DPT是目前最常用的模式。
    # ============================================================
    def _initialize_prediction_heads(self, pred_head_config):
        """
        根据配置初始化预测头。
        设置linear/DPT/DPT+pose三种预测头模式，并创建对应的模型组件。
        """
        # 将图像编码器的patch大小注入预测头配置
        pred_head_config["feature_head"]["patch_size"] = self.encoder.patch_size
        if self.pred_head_type == "linear":
            # Linear模式：直接从最后一层特征做线性映射
            pred_head_config["feature_head"]["input_feature_dim"] = (
                self.info_sharing.dim  # 输入特征维度 = Transformer的输出维度
            )
        elif "dpt" in self.pred_head_type:
            # DPT模式：使用多尺度特征做密集预测，效果更好
            if self.use_encoder_features_for_dpt:
                # 使用编码器特征 + Transformer的2层中间特征 + 最终特征 = 4层多尺度特征
                pred_head_config["feature_head"]["input_feature_dims"] = [
                    self.encoder.enc_embed_dim  # 第1层：编码器原始特征
                ] + [self.info_sharing.dim] * 3  # 第2-4层：Transformer中间层和最终层特征
            else:
                # 使用Transformer的3层中间特征 + 最终特征 = 4层多尺度特征
                pred_head_config["feature_head"]["input_feature_dims"] = [
                    self.info_sharing.dim
                ] * 4
            # 设置DPT回归处理器的输入维度
            pred_head_config["regressor_head"]["input_feature_dim"] = pred_head_config[
                "feature_head"
            ]["feature_dim"]  # 从DPT特征头的输出维度获取
            # 如果需要位姿预测，额外配置位姿预测头
            if "pose" in self.pred_head_type:
                pred_head_config["pose_head"]["patch_size"] = self.encoder.patch_size  # 位姿头需要patch大小信息
                pred_head_config["pose_head"]["input_feature_dim"] = (
                    self.info_sharing.dim  # 位姿头的输入维度 = Transformer输出维度
                )
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )
        # Scale头的输入维度 = Transformer输出维度
        pred_head_config["scale_head"]["input_feature_dim"] = self.info_sharing.dim

        # 实例化预测头模块
        if self.pred_head_type == "linear":
            # 线性预测头：简单但快速
            self.dense_head = LinearFeature(**pred_head_config["feature_head"])
        elif "dpt" in self.pred_head_type:
            # DPT预测头：由特征提取层和回归处理层串联组成
            self.dpt_feature_head = DPTFeature(**pred_head_config["feature_head"])  # DPT特征层：多尺度特征融合与上采样
            self.dpt_regressor_head = DPTRegressionProcessor(  # DPT回归层：从融合特征回归出最终通道数
                **pred_head_config["regressor_head"]
            )
            self.dense_head = nn.Sequential(  # 将两层封装为顺序执行模块
                self.dpt_feature_head, self.dpt_regressor_head
            )
            # 如果需要位姿预测，初始化位姿预测头
            if "pose" in self.pred_head_type:
                self.pose_head = PoseHead(**pred_head_config["pose_head"])  # 位姿头：预测相机平移和旋转四元数
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )
        # 尺度预测头：从scale token的特征预测度量缩放因子
        self.scale_head = MLPHead(**pred_head_config["scale_head"])

    # ============================================================
    # 【重要代码段】初始化输出适配器（Adaptors）
    # 重要性：适配器是预测头原始输出和最终场景表示之间的桥梁。
    # 它负责将预测头输出的原始通道分割为具有物理含义的量（如3D点、光线方向、深度、置信度、掩码等）。
    # scene_rep_type 决定了模型输出的场景表示类型，直接影响forward()中的后处理逻辑。
    # 支持的表示类型包括：
    #   - pointmap: 直接预测世界坐标系下的3D点
    #   - raymap+depth: 预测光线原点+方向+沿光线深度
    #   - raydirs+depth+pose: 预测光线方向+深度+相机位姿（分解式表示）
    #   - campointmap+pose: 预测相机坐标系下的点云+相机位姿
    #   - pointmap+raydirs+depth+pose: 同时预测世界点云和分解式表示
    # 每种类型可选附加 +confidence（置信度）和/或 +mask（掩码）
    # ============================================================
    def _initialize_adaptors(self, pred_head_config):
        """
        根据配置初始化输出适配器。
        为不同的场景表示类型设置相应的适配器。
        """
        # --- 点云类适配器 ---
        if pred_head_config["adaptor_type"] == "pointmap":
            self.dense_adaptor = PointMapAdaptor(**pred_head_config["adaptor"])  # 纯点云适配器：输出(x,y,z)
            self.scene_rep_type = "pointmap"
        elif pred_head_config["adaptor_type"] == "pointmap+confidence":
            self.dense_adaptor = PointMapWithConfidenceAdaptor(  # 点云+置信度：输出(x,y,z) + 置信度
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "pointmap+confidence"
        elif pred_head_config["adaptor_type"] == "pointmap+mask":
            self.dense_adaptor = PointMapWithMaskAdaptor(**pred_head_config["adaptor"])  # 点云+掩码
            self.scene_rep_type = "pointmap+mask"
        elif pred_head_config["adaptor_type"] == "pointmap+confidence+mask":
            self.dense_adaptor = PointMapWithConfidenceAndMaskAdaptor(  # 点云+置信度+掩码
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "pointmap+confidence+mask"
        # --- 光线图+深度类适配器 ---
        elif pred_head_config["adaptor_type"] == "raymap+depth":
            self.dense_adaptor = RayMapPlusDepthAdaptor(**pred_head_config["adaptor"])  # 光线图+深度：输出(光线原点, 光线方向, 深度)
            self.scene_rep_type = "raymap+depth"
        elif pred_head_config["adaptor_type"] == "raymap+depth+confidence":
            self.dense_adaptor = RayMapPlusDepthWithConfidenceAdaptor(  # 光线图+深度+置信度
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+confidence"
        elif pred_head_config["adaptor_type"] == "raymap+depth+mask":
            self.dense_adaptor = RayMapPlusDepthWithMaskAdaptor(  # 光线图+深度+掩码
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+mask"
        elif pred_head_config["adaptor_type"] == "raymap+depth+confidence+mask":
            self.dense_adaptor = RayMapPlusDepthWithConfidenceAndMaskAdaptor(  # 光线图+深度+置信度+掩码
                **pred_head_config["adaptor"]
            )
            self.scene_rep_type = "raymap+depth+confidence+mask"
        # --- 光线方向+深度+位姿类适配器（需要dpt+pose预测头） ---
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthAdaptor(  # 密集适配器：从DPT输出中分割光线方向和深度
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(  # 位姿适配器：从位姿头输出中分割平移和四元数
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+confidence":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithConfidenceAdaptor(  # 光线方向+深度+置信度
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+confidence"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithMaskAdaptor(  # 光线方向+深度+掩码
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+mask"
        elif pred_head_config["adaptor_type"] == "raydirs+depth+pose+confidence+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = RayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(  # 光线方向+深度+置信度+掩码
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "raydirs+depth+pose+confidence+mask"
        # --- 相机坐标系点云+位姿类适配器 ---
        elif pred_head_config["adaptor_type"] == "campointmap+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapAdaptor(**pred_head_config["dpt_adaptor"])  # 相机坐标系点云
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(  # 位姿适配器
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+confidence":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithConfidenceAdaptor(  # 相机坐标系点云+置信度
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+confidence"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithMaskAdaptor(  # 相机坐标系点云+掩码
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+mask"
        elif pred_head_config["adaptor_type"] == "campointmap+pose+confidence+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Camera pointmap + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapWithConfidenceAndMaskAdaptor(  # 相机坐标系点云+置信度+掩码
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "campointmap+pose+confidence+mask"
        # --- 世界点云+光线方向+深度+位姿类适配器（最完整的表示） ---
        elif pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose":
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapPlusRayDirectionsPlusDepthAdaptor(  # 世界点云+光线方向+深度
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(  # 位姿适配器
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose"
        elif (
            pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose+confidence"
        ):
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = (
                PointMapPlusRayDirectionsPlusDepthWithConfidenceAdaptor(  # 世界点云+光线方向+深度+置信度
                    **pred_head_config["dpt_adaptor"]
                )
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+confidence"
        elif pred_head_config["adaptor_type"] == "pointmap+raydirs+depth+pose+mask":
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = PointMapPlusRayDirectionsPlusDepthWithMaskAdaptor(  # 世界点云+光线方向+深度+掩码
                **pred_head_config["dpt_adaptor"]
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+mask"
        elif (
            pred_head_config["adaptor_type"]
            == "pointmap+raydirs+depth+pose+confidence+mask"
        ):
            assert self.pred_head_type == "dpt+pose", (
                "Pointmap + ray directions + depth + pose can only be used as scene representation with dpt + pose head."
            )
            self.dense_adaptor = (
                PointMapPlusRayDirectionsPlusDepthWithConfidenceAndMaskAdaptor(  # 最完整的表示：世界点云+光线方向+深度+置信度+掩码
                    **pred_head_config["dpt_adaptor"]
                )
            )
            self.pose_adaptor = CamTranslationPlusQuatsAdaptor(
                **pred_head_config["pose_adaptor"]
            )
            self.scene_rep_type = "pointmap+raydirs+depth+pose+confidence+mask"
        else:
            raise ValueError(
                f"Invalid adaptor_type: {pred_head_config['adaptor_type']}. \
                Valid options: ['pointmap', 'raymap+depth', 'raydirs+depth+pose', 'campointmap+pose', 'pointmap+raydirs+depth+pose' \
                                'pointmap+confidence', 'raymap+depth+confidence', 'raydirs+depth+pose+confidence', 'campointmap+pose+confidence', 'pointmap+raydirs+depth+pose+confidence' \
                                'pointmap+mask', 'raymap+depth+mask', 'raydirs+depth+pose+mask', 'campointmap+pose+mask', 'pointmap+raydirs+depth+pose+mask' \
                                'pointmap+confidence+mask', 'raymap+depth+confidence+mask', 'raydirs+depth+pose+confidence+mask', 'campointmap+pose+confidence+mask', 'pointmap+raydirs+depth+pose+confidence+mask']"
            )
        # 尺度适配器：从scale head的输出中提取度量缩放因子值
        self.scale_adaptor = ScaleAdaptor(**pred_head_config["scale_adaptor"])

    def _load_pretrained_weights(self):
        """
        从检查点文件加载预训练权重。
        支持两种模式：加载全部权重 或 只加载指定子模块的权重（用于微调场景）。
        """
        if self.pretrained_checkpoint_path is not None:  # 只在提供了权重路径时才加载
            if not self.load_specific_pretrained_submodules:
                # 模式一：加载全部权重（完整恢复模型）
                print(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} ..."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)  # 加载检查点文件
                print(self.load_state_dict(ckpt["model"]))  # 将权重加载到模型中，打印匹配信息
            else:
                # 模式二：只加载特定子模块的权重（用于部分迁移学习）
                print(
                    f"Loading pretrained MapAnything weights from {self.pretrained_checkpoint_path} for specific submodules: {self.specific_pretrained_submodules} ..."
                )
                assert self.pred_head_type is not None, (
                    "Specific submodules to load cannot be None."
                )
                ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)  # 加载检查点文件
                filtered_ckpt = {}  # 存储筛选后的权重
                for ckpt_key, ckpt_value in ckpt["model"].items():  # 遍历所有权重键值对
                    for submodule in self.specific_pretrained_submodules:  # 检查是否属于指定的子模块
                        if ckpt_key.startswith(submodule):  # 如果权重名以子模块名开头则保留
                            filtered_ckpt[ckpt_key] = ckpt_value
                print(self.load_state_dict(filtered_ckpt, strict=False))  # strict=False允许不完全匹配

    # ============================================================
    # 【重要代码段】多视图图像编码
    # 重要性：将所有视图的图像拼接成一个大batch，一次性通过图像编码器（如DINOv2 ViT）
    # 进行编码。这种batch处理方式比逐视图编码效率更高。
    # 编码器输出的特征是后续所有处理步骤的基础。
    # ============================================================
    def _encode_n_views(self, views):
        """
        在单次前向传播中编码所有输入视图的图像。
        假设所有视图具有相同的图像尺寸、batch大小和数据归一化类型。

        返回：(各视图的编码特征列表, 各视图的register token列表)
        """
        num_views = len(views)  # 获取视图数量
        data_norm_type = views[0]["data_norm_type"][0]  # 获取数据归一化类型（如ImageNet均值/标准差）
        imgs_list = [view["img"] for view in views]  # 提取每个视图的图像张量
        all_imgs_across_views = torch.cat(imgs_list, dim=0)  # 将所有视图的图像沿batch维度拼接：(B*V, C, H, W)
        encoder_input = ViTEncoderInput(  # 构造ViT编码器的输入数据结构
            image=all_imgs_across_views, data_norm_type=data_norm_type
        )
        encoder_output = self.encoder(encoder_input)  # 通过图像编码器进行前向传播
        all_encoder_features_across_views = encoder_output.features.chunk(  # 将编码特征按视图数量分割回各个视图
            num_views, dim=0  # 每个元素形状: (B, C, H', W')，其中H'=H/patch_size, W'=W/patch_size
        )
        all_encoder_registers_across_views = None  # 初始化register token为None
        if (
            self.use_register_tokens_from_encoder  # 如果配置了使用register token
            and encoder_output.registers is not None  # 且编码器确实输出了register token
        ):
            all_encoder_registers_across_views = encoder_output.registers.chunk(  # 将register token也按视图分割
                num_views, dim=0
            )

        return all_encoder_features_across_views, all_encoder_registers_across_views

    # ============================================================
    # 【重要代码段】计算所有视图相对于参考视图0的位姿
    # 重要性：多视图3D重建需要知道各视图之间的相对位姿关系。
    # 该方法将所有视图的绝对位姿转换为相对于第一个视图（view 0）的相对位姿。
    # 这种相对位姿表示使模型不依赖于全局坐标系，增强了泛化能力。
    # ============================================================
    def _compute_pose_quats_and_trans_for_across_views_in_ref_view(
        self,
        views,  # 视图列表
        num_views,  # 视图数量
        device,  # 计算设备
        dtype,  # 数据类型
        batch_size_per_view,  # 每个视图的batch大小
        per_sample_cam_input_mask,  # 每个样本的相机输入掩码 (B*V,)
    ):
        """
        计算所有视图在参考视图0坐标系下的位姿（四元数和平移）。
        对于camera_input_mask为False或未提供位姿的视图，返回单位位姿。

        返回：(所有视图的四元数, 所有视图的平移, 更新后的掩码)
        """
        # 收集所有有位姿信息的视图的四元数和平移
        pose_quats_non_ref_views = []  # 各视图自身的四元数
        pose_trans_non_ref_views = []  # 各视图自身的平移
        pose_quats_ref_view_0 = []  # 对应的参考视图0的四元数
        pose_trans_ref_view_0 = []  # 对应的参考视图0的平移
        for view_idx in range(num_views):
            # 获取当前视图对应的样本掩码
            per_sample_cam_input_mask_for_curr_view = per_sample_cam_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ]
            if (
                "camera_pose_quats" in views[view_idx]  # 检查当前视图是否提供了位姿四元数
                and "camera_pose_trans" in views[view_idx]  # 检查当前视图是否提供了位姿平移
                and per_sample_cam_input_mask_for_curr_view.any()  # 检查是否有至少一个样本需要处理
            ):
                # 获取当前视图中有效样本的位姿四元数和平移
                cam_pose_quats = views[view_idx]["camera_pose_quats"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                cam_pose_trans = views[view_idx]["camera_pose_trans"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                pose_quats_non_ref_views.append(cam_pose_quats)  # 添加到当前视图列表
                pose_trans_non_ref_views.append(cam_pose_trans)
                # 获取参考视图0中对应有效样本的位姿
                cam_pose_quats = views[0]["camera_pose_quats"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                cam_pose_trans = views[0]["camera_pose_trans"][
                    per_sample_cam_input_mask_for_curr_view
                ]
                pose_quats_ref_view_0.append(cam_pose_quats)  # 添加到参考视图列表
                pose_trans_ref_view_0.append(cam_pose_trans)
            else:
                # 如果位姿信息不可用，将该视图的掩码全部设为False
                per_sample_cam_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False

        # 将所有视图的位姿初始化为单位位姿（无旋转、无平移）
        pose_quats_across_views = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device  # 单位四元数 (q_x, q_y, q_z, q_w)
        ).repeat(batch_size_per_view * num_views, 1)  # 扩展到所有样本
        pose_trans_across_views = torch.zeros(  # 零平移
            (batch_size_per_view * num_views, 3), dtype=dtype, device=device
        )

        # 如果有需要处理的位姿，计算相对位姿
        if len(pose_quats_non_ref_views) > 0:
            # 将所有视图的位姿拼接为单个张量
            pose_quats_non_ref_views = torch.cat(pose_quats_non_ref_views, dim=0)
            pose_trans_non_ref_views = torch.cat(pose_trans_non_ref_views, dim=0)
            pose_quats_ref_view_0 = torch.cat(pose_quats_ref_view_0, dim=0)
            pose_trans_ref_view_0 = torch.cat(pose_trans_ref_view_0, dim=0)

            # 核心计算：将各视图位姿从世界坐标系转换到参考视图0的坐标系
            # 公式：T_rel = T_ref^{-1} * T_curr，得到视图curr相对于视图ref的相对变换
            (
                pose_quats_non_ref_views_in_ref_view_0,
                pose_trans_non_ref_views_in_ref_view_0,
            ) = transform_pose_using_quats_and_trans_2_to_1(
                pose_quats_ref_view_0,  # 参考视图0的四元数
                pose_trans_ref_view_0,  # 参考视图0的平移
                pose_quats_non_ref_views,  # 当前视图的四元数
                pose_trans_non_ref_views,  # 当前视图的平移
            )

            # 将计算得到的相对位姿填入对应位置（仅更新掩码为True的样本）
            pose_quats_across_views[per_sample_cam_input_mask] = (
                pose_quats_non_ref_views_in_ref_view_0.to(dtype=dtype)
            )
            pose_trans_across_views[per_sample_cam_input_mask] = (
                pose_trans_non_ref_views_in_ref_view_0.to(dtype=dtype)
            )

        return (
            pose_quats_across_views,  # 所有视图的相对四元数 (B*V, 4)
            pose_trans_across_views,  # 所有视图的相对平移 (B*V, 3)
            per_sample_cam_input_mask,  # 更新后的掩码
        )

    def _encode_and_fuse_ray_dirs(
        self,
        views,  # 视图列表
        num_views,  # 视图数量
        batch_size_per_view,  # 每个视图的batch大小
        all_encoder_features_across_views,  # 所有视图的编码特征 (B*V, C, H', W')
        per_sample_ray_dirs_input_mask,  # 光线方向输入掩码 (B*V,)
    ):
        """
        编码所有视图的光线方向并与图像编码特征融合。
        光线方向编码了相机的内参信息（焦距、主点等），帮助模型理解像素到3D空间的映射关系。
        """
        _, _, height, width = views[0]["img"].shape  # 获取图像的高度和宽度

        # 收集所有视图的光线方向数据
        ray_dirs_list = []
        for view_idx in range(num_views):
            # 获取当前视图的光线方向输入掩码
            per_sample_ray_dirs_input_mask_for_curr_view = (
                per_sample_ray_dirs_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ]
            )
            # 初始化当前视图的光线方向为全零（对于没有光线方向的样本保持为零）
            ray_dirs_for_curr_view = torch.zeros(
                (batch_size_per_view, height, width, 3),
                dtype=all_encoder_features_across_views.dtype,
                device=all_encoder_features_across_views.device,
            )
            if (
                "ray_directions_cam" in views[view_idx]  # 检查当前视图是否提供了光线方向
                and per_sample_ray_dirs_input_mask_for_curr_view.any()  # 检查掩码是否有有效样本
            ):
                # 将有效样本的光线方向填入对应位置
                ray_dirs_for_curr_view[per_sample_ray_dirs_input_mask_for_curr_view] = (
                    views[view_idx]["ray_directions_cam"][
                        per_sample_ray_dirs_input_mask_for_curr_view
                    ]
                )
            else:
                # 没有光线方向数据时，将掩码设为False
                per_sample_ray_dirs_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False
            ray_dirs_list.append(ray_dirs_for_curr_view)

        # 拼接所有视图的光线方向并调整维度顺序
        ray_dirs = torch.cat(ray_dirs_list, dim=0)  # (B*V, H, W, 3)
        ray_dirs = ray_dirs.permute(0, 3, 1, 2).contiguous()  # 转为通道优先: (B*V, 3, H, W)

        # 通过光线方向编码器生成特征
        ray_dirs_features_across_views = self.ray_dirs_encoder(
            ViTEncoderNonImageInput(data=ray_dirs)
        ).features  # 输出形状与图像编码特征相同: (B*V, C, H', W')

        # 将掩码为False的样本的光线方向特征置零（确保无效样本不影响融合）
        ray_dirs_features_across_views = (
            ray_dirs_features_across_views
            * per_sample_ray_dirs_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 广播掩码到 (B*V, 1, 1, 1)
        )
        # 将光线方向特征通过逐元素相加融合到图像编码特征中
        all_encoder_features_across_views = (
            all_encoder_features_across_views + ray_dirs_features_across_views
        )

        return all_encoder_features_across_views

    # ============================================================
    # 【重要代码段】深度编码与融合
    # 重要性：深度信息是3D重建的关键先验。该方法实现了：
    # 1. 可选的深度稀疏化（数据增强，模拟真实场景中的稀疏深度传感器）
    # 2. 深度归一化（每视图归一化，使模型对不同尺度的深度输入具有鲁棒性）
    # 3. 对数缩放（更好地处理深度值的长尾分布）
    # 4. 深度尺度因子的独立编码（用于区分度量尺度和相对尺度的深度）
    # ============================================================
    def _encode_and_fuse_depths(
        self,
        views,  # 视图列表
        num_views,  # 视图数量
        batch_size_per_view,  # 每个视图的batch大小
        all_encoder_features_across_views,  # 所有视图的编码特征 (B*V, C, H', W')
        per_sample_depth_input_mask,  # 深度输入掩码 (B*V,)
    ):
        """
        编码所有视图的深度并与图像编码特征融合。
        包含深度归一化、对数缩放、稀疏采样和尺度因子编码等处理步骤。
        """
        device = all_encoder_features_across_views.device  # 获取当前设备
        _, _, height, width = views[0]["img"].shape  # 获取图像尺寸

        # 训练时的数据增强：随机决定是否使用稀疏深度（模拟稀疏深度传感器如LiDAR）
        if torch.rand(1) < self.geometric_input_config["sparse_depth_prob"]:
            use_sparse_depth = True  # 使用稀疏深度
        else:
            use_sparse_depth = False  # 使用密集深度

        # 遍历所有视图，收集深度数据
        depth_list = []  # 归一化后的深度
        depth_norm_factors_list = []  # 深度归一化因子（用于恢复原始尺度）
        metric_scale_depth_mask_list = []  # 度量尺度掩码（标识哪些深度是绝对度量的）
        for view_idx in range(num_views):
            # 获取当前视图的深度输入掩码
            per_sample_depth_input_mask_for_curr_view = per_sample_depth_input_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ]
            # 初始化当前视图的深度为全零
            depth_for_curr_view = torch.zeros(
                (batch_size_per_view, height, width, 1),
                dtype=all_encoder_features_across_views.dtype,
                device=device,
            )
            # 初始化当前视图的归一化因子为全零
            depth_norm_factor_for_curr_view = torch.zeros(
                (batch_size_per_view),
                dtype=all_encoder_features_across_views.dtype,
                device=device,
            )
            # 初始化当前视图的度量尺度掩码为全False
            metric_scale_mask_for_curr_view = torch.zeros(
                (batch_size_per_view),
                dtype=torch.bool,
                device=device,
            )
            if (
                "depth_along_ray" in views[view_idx]  # 检查是否提供了沿光线深度
            ) and per_sample_depth_input_mask_for_curr_view.any():
                # 获取当前视图中有效样本的深度值
                depth_for_curr_view_input = views[view_idx]["depth_along_ray"][
                    per_sample_depth_input_mask_for_curr_view
                ]
                # 获取度量尺度掩码（标识深度是否为绝对度量尺度，如以米为单位）
                if "is_metric_scale" in views[view_idx]:
                    metric_scale_mask = views[view_idx]["is_metric_scale"][
                        per_sample_depth_input_mask_for_curr_view
                    ]
                else:
                    metric_scale_mask = torch.zeros(  # 未提供时默认为非度量尺度
                        depth_for_curr_view_input.shape[0],
                        dtype=torch.bool,
                        device=device,
                    )
                # 训练时的数据增强：以一定概率关闭度量尺度标志
                # 这迫使模型在没有绝对尺度信息时也能工作
                depth_scale_norm_all_mask = (
                    torch.rand(metric_scale_mask.shape[0])
                    < self.geometric_input_config["depth_scale_norm_all_prob"]
                )
                if depth_scale_norm_all_mask.any():
                    metric_scale_mask[depth_scale_norm_all_mask] = False  # 随机关闭部分样本的度量尺度标志
                # 将度量尺度掩码填入对应位置
                metric_scale_mask_for_curr_view[
                    per_sample_depth_input_mask_for_curr_view
                ] = metric_scale_mask
                # 如果需要稀疏深度，随机去除一定比例的深度像素
                if use_sparse_depth:
                    sparsification_mask = torch.ones_like(  # 创建全1掩码
                        depth_for_curr_view_input, device=device
                    )
                    valid_pixel_mask = depth_for_curr_view_input > 0  # 有效像素掩码（深度>0）
                    num_valid_pixels = valid_pixel_mask.sum().item()  # 统计有效像素数
                    # 计算需要移除的像素数
                    num_to_zero = int(
                        num_valid_pixels
                        * self.geometric_input_config["sparsification_removal_percent"]
                    )
                    if num_to_zero > 0:
                        valid_indices = valid_pixel_mask.nonzero(as_tuple=True)  # 获取有效像素的索引
                        indices_to_zero = torch.randperm(num_valid_pixels)[:num_to_zero]  # 随机选择要清零的索引
                        # 在掩码中将选定的像素设为0
                        sparsification_mask[
                            valid_indices[0][indices_to_zero],
                            valid_indices[1][indices_to_zero],
                            valid_indices[2][indices_to_zero],
                            valid_indices[3][indices_to_zero],
                        ] = 0
                    # 应用稀疏化掩码，将选定像素的深度置零
                    depth_for_curr_view_input = (
                        depth_for_curr_view_input * sparsification_mask
                    )
                # 对深度进行归一化：使用非零像素的统计信息将深度缩放到标准范围
                scaled_depth_for_curr_view_input, depth_norm_factor = (
                    normalize_depth_using_non_zero_pixels(
                        depth_for_curr_view_input, return_norm_factor=True  # 同时返回归一化因子，后续用于尺度编码
                    )
                )
                # 将归一化后的深度和归一化因子填入对应位置
                depth_for_curr_view[per_sample_depth_input_mask_for_curr_view] = (
                    scaled_depth_for_curr_view_input
                )
                depth_norm_factor_for_curr_view[
                    per_sample_depth_input_mask_for_curr_view
                ] = depth_norm_factor
            else:
                # 深度数据不可用时，将掩码设为False
                per_sample_depth_input_mask[
                    view_idx * batch_size_per_view : (view_idx + 1)
                    * batch_size_per_view
                ] = False
            # 将当前视图的深度、归一化因子和度量尺度掩码添加到列表
            depth_list.append(depth_for_curr_view)
            depth_norm_factors_list.append(depth_norm_factor_for_curr_view)
            metric_scale_depth_mask_list.append(metric_scale_mask_for_curr_view)

        # 拼接所有视图的深度并应用对数缩放
        depths = torch.cat(depth_list, dim=0)  # (B*V, H, W, 1)
        depths = apply_log_to_norm(
            depths  # 对归一化后的深度值取对数——对数缩放可以更好地处理深度值的长尾分布
        )
        depths = depths.permute(0, 3, 1, 2).contiguous()  # 转为通道优先: (B*V, 1, H, W)
        # 通过深度编码器将深度编码为特征
        depth_features_across_views = self.depth_encoder(
            ViTEncoderNonImageInput(data=depths)
        ).features  # 输出: (B*V, C, H', W')
        # 将掩码为False的样本的深度特征置零
        depth_features_across_views = (
            depth_features_across_views
            * per_sample_depth_input_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )

        # 编码深度归一化因子（尺度信息）
        depth_norm_factors = torch.cat(depth_norm_factors_list, dim=0)  # (B*V,) 所有视图的归一化因子
        log_depth_norm_factors = torch.log(depth_norm_factors + 1e-8)  # 取对数（+eps防止log(0)）
        depth_scale_features_across_views = self.depth_scale_encoder(  # 通过尺度编码器编码
            EncoderGlobalRepInput(data=log_depth_norm_factors.unsqueeze(-1))
        ).features  # 全局特征: (B*V, C)
        # 将掩码为False的样本的尺度特征置零
        depth_scale_features_across_views = (
            depth_scale_features_across_views
            * per_sample_depth_input_mask.unsqueeze(-1)
        )
        # 拼接所有视图的度量尺度掩码
        metric_scale_depth_mask = torch.cat(
            metric_scale_depth_mask_list, dim=0
        )  # (B*V,)
        # 非度量尺度的样本不提供尺度编码（将其置零）
        # 仅在度量尺度样本上保留尺度信息，避免误导模型
        depth_scale_features_across_views = (
            depth_scale_features_across_views * metric_scale_depth_mask.unsqueeze(-1)
        )

        # 将深度特征和深度尺度特征通过逐元素相加融合到图像编码特征中
        # 尺度特征是全局的(B*V, C)，需要广播到空间维度(B*V, C, 1, 1)
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + depth_features_across_views  # 密集深度特征（空间信息）
            + depth_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)  # 全局尺度特征（广播到每个空间位置）
        )

        return all_encoder_features_across_views

    def _encode_and_fuse_cam_quats_and_trans(
        self,
        views,  # 视图列表
        num_views,  # 视图数量
        batch_size_per_view,  # 每个视图的batch大小
        all_encoder_features_across_views,  # 所有视图的编码特征 (B*V, C, H', W')
        pose_quats_across_views,  # 相对位姿四元数 (B*V, 4)，在参考视图0坐标系下
        pose_trans_across_views,  # 相对位姿平移 (B*V, 3)，在参考视图0坐标系下
        per_sample_cam_input_mask,  # 相机输入掩码 (B*V,)
    ):
        """
        编码所有视图的相机旋转（四元数）和平移，并与图像编码特征融合。
        旋转和平移分别通过独立的编码器编码，平移还需要额外编码其尺度因子。
        """
        # 通过相机旋转编码器编码四元数
        pose_quats_features_across_views = self.cam_rot_encoder(
            EncoderGlobalRepInput(data=pose_quats_across_views)  # 输入: (B*V, 4) 四元数
        ).features  # 输出: (B*V, C) 旋转特征
        # 将掩码为False的样本的旋转特征置零
        pose_quats_features_across_views = (
            pose_quats_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # 收集所有视图的度量尺度掩码（标识平移是否为绝对度量尺度）
        device = all_encoder_features_across_views.device
        metric_scale_pose_trans_mask = torch.zeros(
            (batch_size_per_view * num_views), dtype=torch.bool, device=device
        )
        for view_idx in range(num_views):
            if "is_metric_scale" in views[view_idx]:
                metric_scale_mask = views[view_idx]["is_metric_scale"]  # 获取度量尺度标志
            else:
                metric_scale_mask = torch.zeros(  # 未提供时默认为非度量尺度
                    batch_size_per_view, dtype=torch.bool, device=device
                )
            metric_scale_pose_trans_mask[
                view_idx * batch_size_per_view : (view_idx + 1) * batch_size_per_view
            ] = metric_scale_mask

        # 训练时的数据增强：以一定概率关闭度量尺度标志，增强模型对未知尺度的鲁棒性
        pose_norm_all_mask = (
            torch.rand(batch_size_per_view * num_views)
            < self.geometric_input_config["pose_scale_norm_all_prob"]
        )
        if pose_norm_all_mask.any():
            metric_scale_pose_trans_mask[pose_norm_all_mask] = False  # 随机关闭部分样本的度量尺度标志

        # 对平移向量进行跨视图归一化（使不同视图的平移处于统一尺度）
        pose_trans_across_views = torch.split(
            pose_trans_across_views, batch_size_per_view, dim=0  # 按视图拆分
        )  # 结果: num_views个 (B, 3)
        pose_trans_across_views = torch.stack(
            pose_trans_across_views, dim=1  # 沿新维度堆叠
        )  # 形状: (B, num_views, 3)
        scaled_pose_trans_across_views, pose_trans_norm_factors = (
            normalize_pose_translations(  # 归一化平移向量，同时返回归一化因子
                pose_trans_across_views, return_norm_factor=True
            )
        )

        # 将归一化后的平移恢复为 (B*V, 3) 的形状
        scaled_pose_trans_across_views = scaled_pose_trans_across_views.unbind(
            dim=1  # 沿视图维度拆分回列表
        )
        scaled_pose_trans_across_views = torch.cat(
            scaled_pose_trans_across_views, dim=0  # 拼接回 (B*V, 3)
        )
        # 将归一化因子扩展到所有视图: (B,) -> (B*V, 1)
        pose_trans_norm_factors_across_views = pose_trans_norm_factors.unsqueeze(
            -1
        ).repeat(num_views, 1)

        # 通过相机平移编码器编码归一化后的平移
        pose_trans_features_across_views = self.cam_trans_encoder(
            EncoderGlobalRepInput(data=scaled_pose_trans_across_views)  # 输入: (B*V, 3) 归一化平移
        ).features  # 输出: (B*V, C) 平移特征
        # 将掩码为False的样本的平移特征置零
        pose_trans_features_across_views = (
            pose_trans_features_across_views * per_sample_cam_input_mask.unsqueeze(-1)
        )

        # 编码平移归一化因子（取对数后编码，包含绝对尺度信息）
        log_pose_trans_norm_factors_across_views = torch.log(
            pose_trans_norm_factors_across_views + 1e-8  # +eps防止log(0)
        )
        pose_trans_scale_features_across_views = self.cam_trans_scale_encoder(  # 通过平移尺度编码器编码
            EncoderGlobalRepInput(data=log_pose_trans_norm_factors_across_views)
        ).features  # 全局特征: (B*V, C)
        # 将掩码为False的样本的尺度特征置零
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * per_sample_cam_input_mask.unsqueeze(-1)
        )
        # 非度量尺度的样本不提供尺度编码（尺度编码仅用于度量尺度样本）
        pose_trans_scale_features_across_views = (
            pose_trans_scale_features_across_views
            * metric_scale_pose_trans_mask.unsqueeze(-1)
        )

        # 将旋转特征、平移特征和平移尺度特征通过逐元素相加融合到图像编码特征中
        # 这三个都是全局特征 (B*V, C)，需要广播到空间维度 (B*V, C, 1, 1)
        all_encoder_features_across_views = (
            all_encoder_features_across_views
            + pose_quats_features_across_views.unsqueeze(-1).unsqueeze(-1)  # 旋转特征
            + pose_trans_features_across_views.unsqueeze(-1).unsqueeze(-1)  # 平移特征
            + pose_trans_scale_features_across_views.unsqueeze(-1).unsqueeze(-1)  # 平移尺度特征
        )

        return all_encoder_features_across_views

    # ============================================================
    # 【重要代码段】几何模态编码与融合的总调度方法
    # 重要性：这是整个几何条件注入的入口方法。它协调了三种几何先验（光线方向、深度、相机位姿）
    # 的编码和融合过程，并通过随机掩码机制实现训练时的数据增强。
    # 掩码机制使模型能够在推理时灵活处理不同组合的几何输入（全部/部分/无几何输入）。
    # ============================================================
    def _encode_and_fuse_optional_geometric_inputs(
        self, views, all_encoder_features_across_views_list
    ):
        """
        编码所有可选的几何模态输入，并与图像编码特征融合。
        通过多层随机掩码机制实现训练时的灵活几何输入dropout。
        """
        num_views = len(views)  # 视图数量
        batch_size_per_view, _, _, _ = views[0]["img"].shape  # 每个视图的batch大小
        device = all_encoder_features_across_views_list[0].device  # 获取设备
        dtype = all_encoder_features_across_views_list[0].dtype  # 获取数据类型
        # 将所有视图的编码特征拼接为单个张量 (B*V, C, H', W')
        all_encoder_features_across_views = torch.cat(
            all_encoder_features_across_views_list, dim=0
        )

        # --- 多层掩码机制（训练时的数据增强策略） ---

        # 第1层：整体几何输入掩码——以overall_prob概率决定是否使用几何输入
        # 同一batch中的所有视图共享相同的掩码（保持视图间一致性）
        overall_geometric_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["overall_prob"]
        )
        overall_geometric_input_mask = overall_geometric_input_mask.repeat(num_views)  # 扩展到所有视图

        # 第2层：每样本dropout掩码——在整体掩码的基础上进一步随机丢弃个别样本
        per_sample_geometric_input_mask = torch.rand(
            batch_size_per_view * num_views, device=device
        ) < (1 - self.geometric_input_config["dropout_prob"])
        per_sample_geometric_input_mask = (
            per_sample_geometric_input_mask & overall_geometric_input_mask  # 与整体掩码取交集
        )

        # 第3层：各模态独立掩码——每种几何模态都有独立的使用概率
        # 光线方向掩码
        per_sample_ray_dirs_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["ray_dirs_prob"]
        )
        per_sample_ray_dirs_input_mask = per_sample_ray_dirs_input_mask.repeat(
            num_views
        )
        per_sample_ray_dirs_input_mask = (
            per_sample_ray_dirs_input_mask & per_sample_geometric_input_mask  # 与样本掩码取交集
        )

        # 深度掩码
        per_sample_depth_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["depth_prob"]
        )
        per_sample_depth_input_mask = per_sample_depth_input_mask.repeat(num_views)
        per_sample_depth_input_mask = (
            per_sample_depth_input_mask & per_sample_geometric_input_mask
        )

        # 相机位姿掩码
        per_sample_cam_input_mask = (
            torch.rand(batch_size_per_view, device=device)
            < self.geometric_input_config["cam_prob"]
        )
        per_sample_cam_input_mask = per_sample_cam_input_mask.repeat(num_views)
        per_sample_cam_input_mask = (
            per_sample_cam_input_mask & per_sample_geometric_input_mask
        )

        # --- 依次编码和融合各几何模态 ---

        # 步骤1：计算所有视图相对于参考视图0的位姿
        pose_quats_across_views, pose_trans_across_views, per_sample_cam_input_mask = (
            self._compute_pose_quats_and_trans_for_across_views_in_ref_view(
                views,
                num_views,
                device,
                dtype,
                batch_size_per_view,
                per_sample_cam_input_mask,
            )
        )

        # ===== 验证代码：保存两种方式变换到世界坐标系的点云PLY =====
        if False:
            import os, time
            import numpy as np
            from scipy.spatial.transform import Rotation as _Rotation

            _VIEW_COLORS = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (0, 255, 255), (255, 0, 255),
                (255, 128, 0), (128, 0, 255), (0, 128, 255),
                (128, 255, 0), (255, 0, 128), (0, 255, 128),
            ]
            save_dir = "/drobotics-ailab/bohao.zhang/Projects/map-anything/debug_by_vis"
            os.makedirs(save_dir, exist_ok=True)
            ts = int(time.time())

            def _save_ply(path, points, colors):
                n = len(points)
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {n}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n"
                )
                ply_dtype = np.dtype([
                    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                    ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ])
                verts = np.empty(n, dtype=ply_dtype)
                verts['x'] = points[:, 0]
                verts['y'] = points[:, 1]
                verts['z'] = points[:, 2]
                verts['r'] = colors[:, 0]
                verts['g'] = colors[:, 1]
                verts['b'] = colors[:, 2]
                with open(path, 'wb') as f:
                    f.write(header.encode('ascii'))
                    f.write(verts.tobytes())

            for b_idx in range(batch_size_per_view):
                world_pts_list = []
                cam2world_pts_list = []
                cam_in_ref0_pts_list = []  # 方式3: pose_*_across_views（相对 ref view0）变换 pts3d_cam
                colors_list = []

                for view_idx in range(num_views):
                    valid_mask = views[view_idx]["valid_mask"][b_idx].detach().cpu().numpy().reshape(-1)

                    # 方式1: 直接拼接 pts3d（已是世界坐标）
                    pts3d_world_np = (
                        views[view_idx]["pts3d"][b_idx]
                        .detach().cpu().float().numpy().reshape(-1, 3)
                    )[valid_mask]

                    # 方式2: pts3d_cam 经 camera_pose_quats + camera_pose_trans 变换
                    pts_cam_np = (
                        views[view_idx]["pts3d_cam"][b_idx]
                        .detach().cpu().float().numpy().reshape(-1, 3)
                    )[valid_mask]
                    quat_np = views[view_idx]["camera_pose_quats"][b_idx].detach().cpu().float().numpy()
                    trans_np = views[view_idx]["camera_pose_trans"][b_idx].detach().cpu().float().numpy()
                    R_c2w = _Rotation.from_quat(quat_np).as_matrix().astype(np.float32)
                    pts_cam2world_np = (pts_cam_np @ R_c2w.T) + trans_np[None, :]

                    # 方式3: 与编码器一致，使用 T_curr_in_ref0 = T_ref^{-1} T_curr（展平索引 view*B + b）
                    flat_pose_idx = view_idx * batch_size_per_view + b_idx
                    quat_across = pose_quats_across_views[flat_pose_idx].detach().cpu().float().numpy()
                    trans_across = pose_trans_across_views[flat_pose_idx].detach().cpu().float().numpy()
                    R_across = _Rotation.from_quat(quat_across).as_matrix().astype(np.float32)
                    pts_cam_in_ref0_np = (pts_cam_np @ R_across.T) + trans_across[None, :]

                    n_valid = len(pts3d_world_np)
                    world_pts_list.append(pts3d_world_np)
                    cam2world_pts_list.append(pts_cam2world_np)
                    cam_in_ref0_pts_list.append(pts_cam_in_ref0_np)

                    vc = _VIEW_COLORS[view_idx % len(_VIEW_COLORS)]
                    colors_view = np.tile(np.array(vc, dtype=np.uint8), (n_valid, 1))
                    colors_list.append(colors_view)

                    # 每帧（每视图）单独保存
                    ply_view_world = os.path.join(
                        save_dir,
                        f"scene{b_idx}_view{view_idx}_pts3d_{ts}.ply",
                    )
                    _save_ply(ply_view_world, pts3d_world_np, colors_view)
                    ply_view_cam2world = os.path.join(
                        save_dir,
                        f"scene{b_idx}_view{view_idx}_cam2world_{ts}.ply",
                    )
                    _save_ply(ply_view_cam2world, pts_cam2world_np, colors_view)
                    ply_view_cam_in_ref0 = os.path.join(
                        save_dir,
                        f"scene{b_idx}_view{view_idx}_cam_in_ref0_across_{ts}.ply",
                    )
                    _save_ply(ply_view_cam_in_ref0, pts_cam_in_ref0_np, colors_view)
                    print(
                        f"[DEBUG] scene{b_idx} view{view_idx}: "
                        f"{n_valid} pts -> {ply_view_world} / {ply_view_cam2world} / {ply_view_cam_in_ref0}"
                    )

                all_world_pts = np.concatenate(world_pts_list, axis=0)
                all_cam2world_pts = np.concatenate(cam2world_pts_list, axis=0)
                all_colors = np.concatenate(colors_list, axis=0)

                ply_path_world = os.path.join(
                    save_dir, f"scene{b_idx}_{num_views}views_{ts}.ply"
                )
                _save_ply(ply_path_world, all_world_pts, all_colors)
                print(f"[DEBUG] Scene {b_idx}: {len(all_world_pts)} pts (pts3d) -> {ply_path_world}")

                ply_path_cam2world = os.path.join(
                    save_dir, f"scene{b_idx}_{num_views}views_cam2world_{ts}.ply"
                )
                _save_ply(ply_path_cam2world, all_cam2world_pts, all_colors)
                print(f"[DEBUG] Scene {b_idx}: {len(all_cam2world_pts)} pts (quats+trans) -> {ply_path_cam2world}")

                all_cam_in_ref0_pts = np.concatenate(cam_in_ref0_pts_list, axis=0)
                ply_path_cam_in_ref0 = os.path.join(
                    save_dir,
                    f"scene{b_idx}_{num_views}views_pts3d_cam_in_ref0_across_{ts}.ply",
                )
                _save_ply(ply_path_cam_in_ref0, all_cam_in_ref0_pts, all_colors)
                print(
                    f"[DEBUG] Scene {b_idx}: {len(all_cam_in_ref0_pts)} pts "
                    f"(pts3d_cam + pose_*_across_views -> ref0) -> {ply_path_cam_in_ref0}"
                )
                break
        # ===== 验证代码结束 =====

        # 步骤2：编码光线方向并融合（编码相机内参信息）
        all_encoder_features_across_views = self._encode_and_fuse_ray_dirs(
            views,
            num_views,
            batch_size_per_view,
            all_encoder_features_across_views,
            per_sample_ray_dirs_input_mask,
        )

        # 步骤3：编码深度并融合（编码场景深度信息）
        # all_encoder_features_across_views = self._encode_and_fuse_depths(
        #     views,
        #     num_views,
        #     batch_size_per_view,
        #     all_encoder_features_across_views,
        #     per_sample_depth_input_mask,
        # )

        # 步骤4：编码相机旋转和平移并融合（编码相机外参信息）
        # all_encoder_features_across_views = self._encode_and_fuse_cam_quats_and_trans(
        #     views,
        #     num_views,
        #     batch_size_per_view,
        #     all_encoder_features_across_views,
        #     pose_quats_across_views,
        #     pose_trans_across_views,
        #     per_sample_cam_input_mask,
        # )

        # 步骤5：对融合后的特征进行LayerNorm归一化（稳定训练）
        # 需要先转为(B*V, H', W', C)以在最后一个维度上做LayerNorm，然后转回来
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 2, 3, 1  # (B*V, C, H', W') -> (B*V, H', W', C)
        ).contiguous()
        all_encoder_features_across_views = self.fusion_norm_layer(  # LayerNorm在特征维度上归一化
            all_encoder_features_across_views
        )
        all_encoder_features_across_views = all_encoder_features_across_views.permute(
            0, 3, 1, 2  # (B*V, H', W', C) -> (B*V, C, H', W')
        ).contiguous()

        # 将拼接的特征按视图拆分回列表
        fused_all_encoder_features_across_views = (
            all_encoder_features_across_views.chunk(num_views, dim=0)
        )

        return fused_all_encoder_features_across_views

    def _compute_adaptive_minibatch_size(
        self,
        memory_safety_factor: float = 0.95,  # 安全系数：使用95%的可用显存，留5%缓冲
    ) -> int:
        """
        根据可用GPU显存自适应计算小批量大小。
        用于内存高效推理模式，避免OOM（显存不足）错误。
        """
        device = self.device

        if device.type == "cuda":
            torch.cuda.empty_cache()  # 清理GPU缓存释放未使用的显存
            available_memory = torch.cuda.mem_get_info()[0]  # 获取可用显存（字节）
            usable_memory = (
                available_memory * memory_safety_factor  # 乘以安全系数避免OOM
            )
        else:
            # 非CUDA设备使用保守的默认值
            print(
                "Non-CUDA device detected. Using conservative default minibatch size of 1 for memory efficient dense prediction head inference."
            )
            return 1

        # 根据可用显存和每个样本的估计显存需求计算小批量大小
        max_estimated_memory_per_sample = (
            680 * 1024 * 1024  # 每个样本约680MB（基于518x518输入的上限估算）
        )
        computed_minibatch_size = int(usable_memory / max_estimated_memory_per_sample)
        if computed_minibatch_size < 1:
            computed_minibatch_size = 1  # 至少处理1个样本

        return computed_minibatch_size

    def downstream_dense_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],  # 密集预测头的输入特征
        img_shape: Tuple[int, int],  # 目标输出的图像尺寸 (H, W)
    ):
        """
        运行下游密集预测头。根据预测头类型（linear/dpt）选择不同的处理流程。
        密集预测头将Transformer特征解码为每像素的场景表示。
        """
        if self.pred_head_type == "linear":
            # Linear模式：直接从最后一层特征做线性映射
            dense_head_outputs = self.dense_head(
                PredictionHeadInput(last_feature=dense_head_inputs)  # 单层特征输入
            )
            dense_final_outputs = self.dense_adaptor(  # 通过适配器将原始通道分割为有意义的输出
                AdaptorInput(
                    adaptor_feature=dense_head_outputs.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )
        elif self.pred_head_type in ["dpt", "dpt+pose"]:
            # DPT模式：使用多尺度特征进行密集预测
            dense_head_outputs = self.dense_head(
                PredictionHeadLayeredInput(
                    list_features=dense_head_inputs,  # 多层特征输入（4层多尺度特征）
                    target_output_shape=img_shape,  # 目标输出分辨率
                )
            )
            dense_final_outputs = self.dense_adaptor(  # 通过适配器分割输出通道
                AdaptorInput(
                    adaptor_feature=dense_head_outputs.decoded_channels,
                    output_shape_hw=img_shape,
                )
            )
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )

        return dense_final_outputs

    # ============================================================
    # 【重要代码段】下游预测头的统一执行入口
    # 重要性：该方法协调密集预测头、位姿预测头和尺度预测头的执行。
    # 它还实现了内存高效推理模式——通过小批量（mini-batch）方式运行DPT预测头，
    # 大幅降低GPU显存需求，使模型能在显存较小的GPU上也能推理。
    # ============================================================
    def downstream_head(
        self,
        dense_head_inputs: Union[torch.Tensor, List[torch.Tensor]],  # 密集预测头的输入特征
        scale_head_inputs: torch.Tensor,  # 尺度预测头的输入特征（来自scale token）
        img_shape: Tuple[int, int],  # 图像尺寸 (H, W)
        memory_efficient_inference: bool = False,  # 是否使用内存高效推理模式
        minibatch_size: int = None,  # 可选的固定小批量大小
    ):
        """
        运行所有预测头（密集预测头、位姿预测头、尺度预测头）并返回结果。
        支持内存高效推理模式（以速度换显存）。
        """
        device = self.device  # 获取当前设备

        # 内存高效推理模式：将密集预测头分成小批量执行，减少显存峰值
        if memory_efficient_inference:
            # 获取总的batch大小
            if self.pred_head_type == "linear":
                batch_size = dense_head_inputs.shape[0]  # Linear模式：从张量获取batch大小
            elif self.pred_head_type in ["dpt", "dpt+pose"]:
                batch_size = dense_head_inputs[0].shape[0]  # DPT模式：从特征列表的第一个元素获取
            else:
                raise ValueError(
                    f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
                )

            # 计算小批量大小和批次数
            if minibatch_size is not None:
                minibatch = minibatch_size  # 使用用户指定的小批量大小
            else:
                minibatch = self._compute_adaptive_minibatch_size()  # 根据可用显存自适应计算
            num_batches = (batch_size + minibatch - 1) // minibatch  # 向上取整计算批次数

            # 逐小批量运行预测
            dense_final_outputs_list = []  # 收集每个小批量的密集预测结果
            pose_final_outputs_list = [] if self.pred_head_type == "dpt+pose" else None  # 收集位姿预测结果
            for batch_idx in range(num_batches):
                start_idx = batch_idx * minibatch  # 当前小批量的起始索引
                end_idx = min((batch_idx + 1) * minibatch, batch_size)  # 当前小批量的结束索引

                # 切片获取当前小批量的输入
                if self.pred_head_type == "linear":
                    dense_head_inputs_batch = dense_head_inputs[start_idx:end_idx]
                elif self.pred_head_type in ["dpt", "dpt+pose"]:
                    dense_head_inputs_batch = [  # 对每层特征都进行切片
                        x[start_idx:end_idx] for x in dense_head_inputs
                    ]
                else:
                    raise ValueError(
                        f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
                    )

                # 运行密集预测头（小批量方式）
                dense_final_outputs_batch = self.downstream_dense_head(
                    dense_head_inputs_batch, img_shape
                )
                dense_final_outputs_list.append(dense_final_outputs_batch)

                # 运行位姿预测头（小批量方式，仅在dpt+pose模式下）
                if self.pred_head_type == "dpt+pose":
                    pose_head_inputs_batch = dense_head_inputs[-1][start_idx:end_idx]  # 位姿头使用最后一层特征
                    pose_head_outputs_batch = self.pose_head(
                        PredictionHeadInput(last_feature=pose_head_inputs_batch)
                    )
                    pose_final_outputs_batch = self.pose_adaptor(  # 通过位姿适配器分割平移和四元数
                        AdaptorInput(
                            adaptor_feature=pose_head_outputs_batch.decoded_channels,
                            output_shape_hw=img_shape,
                        )
                    )
                    pose_final_outputs_list.append(pose_final_outputs_batch)

            # 将所有小批量的密集预测结果拼接为完整结果
            available_keys = dense_final_outputs_batch.__dict__.keys()
            dense_pred_data_dict = {
                key: torch.cat(  # 沿batch维度拼接每个属性
                    [getattr(output, key) for output in dense_final_outputs_list], dim=0
                )
                for key in available_keys
            }
            dense_final_outputs = dense_final_outputs_batch.__class__(  # 用同一类重新构造输出对象
                **dense_pred_data_dict
            )

            # 将所有小批量的位姿预测结果拼接为完整结果
            pose_final_outputs = None
            if self.pred_head_type == "dpt+pose":
                available_keys = pose_final_outputs_batch.__dict__.keys()
                pose_pred_data_dict = {
                    key: torch.cat(
                        [getattr(output, key) for output in pose_final_outputs_list],
                        dim=0,
                    )
                    for key in available_keys
                }
                pose_final_outputs = pose_final_outputs_batch.__class__(
                    **pose_pred_data_dict
                )

            # 清理CUDA缓存以释放显存
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            # 非内存高效模式：一次性处理所有样本（更快但更耗显存）
            dense_final_outputs = self.downstream_dense_head(  # 运行密集预测头
                dense_head_inputs, img_shape
            )

            # 运行位姿预测头（仅在dpt+pose模式下）
            pose_final_outputs = None
            if self.pred_head_type == "dpt+pose":
                pose_head_outputs = self.pose_head(
                    PredictionHeadInput(last_feature=dense_head_inputs[-1])  # 位姿头使用最后一层特征
                )
                pose_final_outputs = self.pose_adaptor(
                    AdaptorInput(
                        adaptor_feature=pose_head_outputs.decoded_channels,
                        output_shape_hw=img_shape,
                    )
                )

        # 尺度预测比较轻量，始终一次性处理
        scale_head_output = self.scale_head(
            PredictionHeadTokenInput(last_feature=scale_head_inputs)  # 输入：scale token的特征
        )
        scale_final_output = self.scale_adaptor(  # 通过尺度适配器提取缩放因子
            AdaptorInput(
                adaptor_feature=scale_head_output.decoded_channels,
                output_shape_hw=img_shape,
            )
        )
        scale_final_output = scale_final_output.value.squeeze(-1)  # (B, 1, 1) -> (B, 1) 去掉多余维度

        # 内存高效模式下清理CUDA缓存
        if memory_efficient_inference and device.type == "cuda":
            torch.cuda.empty_cache()

        return dense_final_outputs, pose_final_outputs, scale_final_output  # 返回密集预测、位姿预测和尺度预测

    # ============================================================
    # 【核心代码段】forward() — 模型的前向传播主流程
    # 重要性：这是MapAnything模型的核心方法，定义了从输入到输出的完整数据流：
    #   1. 图像编码 → 2. 几何特征融合 → 3. 多视图信息共享 → 4. 特征组装 → 5. 预测头解码 → 6. 输出组装
    # 所有的训练和推理都通过这个方法执行。
    # ============================================================
    def forward(self, views, memory_efficient_inference=False, minibatch_size=None):
        """
        前向传播主流程：
        1. 编码N个输入视图的图像
        2. 编码可选几何输入（光线方向、深度、相机旋转、相机平移）
        3. 通过加法和归一化融合图像特征与几何特征
        4. 通过多视图注意力Transformer进行跨视图信息共享
        5. 将Transformer输出的特征通过预测头解码
        6. 返回N个视图的最终输出

        Args:
            views: 视图字典列表，每个字典包含"img"(B,C,H,W)和可选的几何输入
            memory_efficient_inference: 是否使用内存高效推理模式
            minibatch_size: 内存高效推理时的固定小批量大小

        Returns:
            包含所有N个视图最终输出的字典列表
        """
        # 获取图像尺寸、视图数和每个视图的batch大小
        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))  # 图像空间分辨率
        num_views = len(views)  # 视图数量

        # ---- 阶段1：图像编码 ----
        # 将所有视图的图像通过ViT编码器编码为特征
        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self._encode_n_views(views)
        )

        # ---- 阶段2：几何特征融合 ----
        # 编码可选几何输入并与图像特征融合
        # 关闭autocast使用高精度（float32），防止LayerNorm中因特征方差过大产生NaN
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        # ---- 阶段3：准备多视图Transformer输入 ----
        # 将scale token扩展到batch大小：(C,) -> (B, C, 1)
        input_scale_token = (
            self.scale_token.unsqueeze(0)  # (1, C)
            .unsqueeze(-1)  # (1, C, 1)
            .repeat(batch_size_per_view, 1, 1)  # (B, C, 1)
        )

        # 构造多视图Transformer的输入数据结构
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,  # 各视图的融合特征
            additional_input_tokens_per_view=all_encoder_registers_across_views,  # 各视图的register token（可选）
            additional_input_tokens=input_scale_token,  # scale token（所有视图共享）
        )

        # ---- 阶段4：多视图信息共享 ----
        # 通过多视图Transformer让不同视图的特征互相交互
        final_info_sharing_multi_view_feat = None  # 最终层特征
        intermediate_info_sharing_multi_view_feat = None  # 中间层特征（DPT需要）
        if self.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self.info_sharing(info_sharing_input)  # 仅返回最终层
        elif self.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,  # 同时返回中间层和最终层
            ) = self.info_sharing(info_sharing_input)

        # ---- 阶段5：组装预测头输入特征 ----
        if self.pred_head_type == "linear":
            # Linear模式：仅使用最终层特征，拼接所有视图
            dense_head_inputs = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0  # (B*V, C, H', W')
            )
        elif self.pred_head_type in ["dpt", "dpt+pose"]:
            # DPT模式：组装4层多尺度特征供DPT使用
            dense_head_inputs_list = []
            if self.use_encoder_features_for_dpt:
                # 方案A：编码器特征 + 2层中间特征 + 最终特征 = 4层
                stacked_encoder_features = torch.cat(  # 第1层：编码器原始特征
                    all_encoder_features_across_views, dim=0
                )
                dense_head_inputs_list.append(stacked_encoder_features)
                stacked_intermediate_features_1 = torch.cat(  # 第2层：Transformer第1中间层
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                stacked_intermediate_features_2 = torch.cat(  # 第3层：Transformer第2中间层
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                stacked_final_features = torch.cat(  # 第4层：Transformer最终层
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                dense_head_inputs_list.append(stacked_final_features)
            else:
                # 方案B：3层中间特征 + 最终特征 = 4层（不需要编码器特征）
                stacked_intermediate_features_1 = torch.cat(  # 第1层：Transformer第1中间层
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                stacked_intermediate_features_2 = torch.cat(  # 第2层：Transformer第2中间层
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                stacked_intermediate_features_3 = torch.cat(  # 第3层：Transformer第3中间层
                    intermediate_info_sharing_multi_view_feat[2].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_3)
                stacked_final_features = torch.cat(  # 第4层：Transformer最终层
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                dense_head_inputs_list.append(stacked_final_features)
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )

        # ---- 阶段6：运行预测头并组装最终输出 ----
        # 关闭autocast使用高精度（float32），确保预测头输出的精度
        with torch.autocast("cuda", enabled=False):
            # 准备预测头输入
            if self.pred_head_type == "linear":
                dense_head_inputs = dense_head_inputs  # 单张量
            elif self.pred_head_type in ["dpt", "dpt+pose"]:
                dense_head_inputs = dense_head_inputs_list  # 多尺度特征列表
            # 尺度预测头的输入：从Transformer输出中提取scale token的特征
            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features       # 
            )

            # 运行所有预测头（密集预测、位姿预测、尺度预测）
            dense_final_outputs, pose_final_outputs, scale_final_output = (
                self.downstream_head(
                    dense_head_inputs=dense_head_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                    memory_efficient_inference=memory_efficient_inference,
                    minibatch_size=minibatch_size,
                )
            )

            # ---- 阶段7：根据场景表示类型组装最终输出 ----
            # 根据不同的scene_rep_type，将预测头输出转换为统一格式的结果字典

            # === 点云类表示：直接预测世界坐标系下的3D点 ===
            if self.scene_rep_type in [
                "pointmap",
                "pointmap+confidence",
                "pointmap+mask",
                "pointmap+confidence+mask",
            ]:
                output_pts3d = dense_final_outputs.value  # 获取原始预测值 (B*V, 3, H, W)
                output_pts3d = output_pts3d.permute(0, 2, 3, 1).contiguous()  # 转为 (B*V, H, W, 3)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)  # 按视图拆分
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            # pts3d乘以scale_final_output恢复到度量尺度
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "metric_scaling_factor": scale_final_output,
                        }
                    )

            # === 光线图+深度类表示：预测光线原点、方向和沿光线深度 ===
            elif self.scene_rep_type in [
                "raymap+depth",
                "raymap+depth+confidence",
                "raymap+depth+mask",
                "raymap+depth+confidence+mask",
            ]:
                output_scene_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()  # (B*V, H, W, 7) = (光线原点3 + 光线方向3 + 深度1)
                # 将预测值按通道分割为光线原点、方向和深度
                output_ray_origins, output_ray_directions, output_depth_along_ray = (
                    output_scene_rep.split([3, 3, 1], dim=-1)
                )
                # 从光线表示计算3D点：P = O + D * d
                output_pts3d = (
                    output_ray_origins + output_ray_directions * output_depth_along_ray
                )
                # 按视图拆分所有预测量
                output_ray_origins_per_view = output_ray_origins.chunk(num_views, dim=0)
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 点云乘以尺度因子
                            "ray_origins": output_ray_origins_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 光线原点乘以尺度因子
                            "ray_directions": output_ray_directions_per_view[i],  # 光线方向不需要缩放
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 深度乘以尺度因子
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            # === 光线方向+深度+位姿类表示（分解式表示，最灵活） ===
            elif self.scene_rep_type in [
                "raydirs+depth+pose",
                "raydirs+depth+pose+confidence",
                "raydirs+depth+pose+mask",
                "raydirs+depth+pose+confidence+mask",
            ]:
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()  # (B*V, H, W, 4) = (光线方向3 + 深度1)
                # 将密集输出分割为光线方向和沿光线深度
                output_ray_directions, output_depth_along_ray = output_dense_rep.split(
                    [3, 1], dim=-1
                )
                # 从位姿预测头获取相机平移和四元数
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)  # 分割为3D平移和4D四元数
                )
                # 从分解表示（光线方向+深度+位姿）重建世界坐标系下的3D点
                # 公式：P_world = R * (ray_dir * depth) + t
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                # 相机坐标系下的3D点 = 光线方向 * 深度
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # 按视图拆分所有预测量
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 世界点云×尺度
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 相机坐标点云×尺度
                            "ray_directions": output_ray_directions_per_view[i],  # 光线方向（无需缩放）
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 深度×尺度
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,  # 相机平移×尺度
                            "cam_quats": output_cam_quats_per_view[i],  # 相机四元数（无需缩放）
                            "metric_scaling_factor": scale_final_output,
                        }
                    )

            # === 相机坐标系点云+位姿类表示 ===
            elif self.scene_rep_type in [
                "campointmap+pose",
                "campointmap+pose+confidence",
                "campointmap+pose+mask",
                "campointmap+pose+confidence+mask",
            ]:
                # 获取预测的相机坐标系下的3D点云
                output_pts3d_cam = dense_final_outputs.value
                output_pts3d_cam = output_pts3d_cam.permute(0, 2, 3, 1).contiguous()  # (B*V, H, W, 3)
                # 从位姿预测头获取相机平移和四元数
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # 从相机坐标系点云反推光线方向和深度
                output_depth_along_ray = torch.norm(  # 深度 = 点到原点的距离
                    output_pts3d_cam, dim=-1, keepdim=True
                )
                output_ray_directions = output_pts3d_cam / output_depth_along_ray  # 光线方向 = 归一化的点坐标
                # 将相机坐标系点云通过位姿变换到世界坐标系
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                # 按视图拆分所有预测量
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 世界点云×尺度
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 相机坐标点云×尺度
                            "ray_directions": output_ray_directions_per_view[i],  # 光线方向
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 深度×尺度
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,  # 相机平移×尺度
                            "cam_quats": output_cam_quats_per_view[i],  # 相机四元数
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            # === 世界点云+光线方向+深度+位姿类表示（最完整的冗余表示） ===
            elif self.scene_rep_type in [
                "pointmap+raydirs+depth+pose",
                "pointmap+raydirs+depth+pose+confidence",
                "pointmap+raydirs+depth+pose+mask",
                "pointmap+raydirs+depth+pose+confidence+mask",
            ]:
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()  # (B*V, H, W, 7) = (世界点云3 + 光线方向3 + 深度1)
                # 将密集输出分割为世界点云、光线方向和深度
                output_pts3d, output_ray_directions, output_depth_along_ray = (
                    output_dense_rep.split([3, 3, 1], dim=-1)
                )
                # 从位姿预测头获取相机平移和四元数
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # 相机坐标系下的3D点 = 光线方向 * 深度
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # 如果配置了使用分解式预测替代直接点云预测（通常效果更好）
                if self.pred_head_config["adaptor_config"][
                    "use_factored_predictions_for_global_pointmaps"
                ]:
                    # 使用分解式表示重新计算世界点云：P_world = R * (ray_dir * depth) + t
                    output_pts3d = (
                        convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                            output_ray_directions,
                            output_depth_along_ray,
                            output_cam_translations,
                            output_cam_quats,
                        )
                    )
                # 按视图拆分所有预测量
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 世界点云×尺度
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 相机坐标点云×尺度
                            "ray_directions": output_ray_directions_per_view[i],  # 光线方向
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),  # 深度×尺度
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,  # 相机平移×尺度
                            "cam_quats": output_cam_quats_per_view[i],  # 相机四元数
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            else:
                raise ValueError(
                    f"Invalid scene_rep_type: {self.scene_rep_type}. \
                    Valid options: ['pointmap', 'raymap+depth', 'raydirs+depth+pose', 'campointmap+pose', 'pointmap+raydirs+depth+pose' \
                                    'pointmap+confidence', 'raymap+depth+confidence', 'raydirs+depth+pose+confidence', 'campointmap+pose+confidence', 'pointmap+raydirs+depth+pose+confidence' \
                                    'pointmap+mask', 'raymap+depth+mask', 'raydirs+depth+pose+mask', 'campointmap+pose+mask', 'pointmap+raydirs+depth+pose+mask' \
                                    'pointmap+confidence+mask', 'raymap+depth+confidence+mask', 'raydirs+depth+pose+confidence+mask', 'campointmap+pose+confidence+mask', 'pointmap+raydirs+depth+pose+confidence+mask']"
                )

            # ---- 附加输出：置信度（如果场景表示类型包含confidence） ----
            if "confidence" in self.scene_rep_type:
                output_confidences = dense_final_outputs.confidence  # 获取置信度预测 (B*V, 1, H, W)
                output_confidences = (
                    output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()  # 转为 (B*V, H, W)
                )
                output_confidences_per_view = output_confidences.chunk(num_views, dim=0)  # 按视图拆分
                for i in range(num_views):
                    res[i]["conf"] = output_confidences_per_view[i]  # 添加置信度到结果字典

            # ---- 附加输出：非模糊区域掩码（如果场景表示类型包含mask） ----
            if "mask" in self.scene_rep_type:
                output_masks = dense_final_outputs.mask  # 获取掩码预测 (B*V, 1, H, W)
                output_masks = output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous()  # 转为 (B*V, H, W)
                output_masks = output_masks > 0.5  # 以0.5为阈值二值化（1:非模糊区域, 0:模糊区域）
                output_masks_per_view = output_masks.chunk(num_views, dim=0)  # 按视图拆分
                # 同时保存掩码logits（用于训练时计算损失）
                output_mask_logits = dense_final_outputs.logits  # 获取掩码logits (B*V, 1, H, W)
                output_mask_logits = (
                    output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()  # 转为 (B*V, H, W)
                )
                output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)  # 按视图拆分
                for i in range(num_views):
                    res[i]["non_ambiguous_mask"] = output_masks_per_view[i]  # 二值化掩码
                    res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]  # 原始logits

        return res  # 返回所有视图的预测结果列表

    def _configure_geometric_input_config(
        self,
        use_calibration: bool,  # 是否使用标定信息（光线方向/内参）
        use_depth: bool,  # 是否使用深度输入
        use_pose: bool,  # 是否使用相机位姿输入
        use_depth_scale: bool,  # 是否使用深度的度量尺度信息
        use_pose_scale: bool,  # 是否使用位姿的度量尺度信息
    ):
        """
        配置推理时的几何输入概率。
        将训练时使用的随机概率替换为确定性的开/关设置（概率为0或1）。
        """
        # 保存原始配置以便推理后恢复（确保不影响训练时的随机行为）
        if not hasattr(self, "_original_geometric_config"):
            self._original_geometric_config = dict(self.geometric_input_config)

        if not (use_calibration or use_depth or use_pose):
            # 纯图像模式：关闭所有几何输入
            self.geometric_input_config.update(
                {
                    "overall_prob": 0.0,  # 完全不使用几何输入
                    "dropout_prob": 1.0,  # 100%丢弃
                    "ray_dirs_prob": 0.0,
                    "depth_prob": 0.0,
                    "cam_prob": 0.0,
                    "sparse_depth_prob": 0.0,
                    "depth_scale_norm_all_prob": 0.0,
                    "pose_scale_norm_all_prob": 0.0,
                }
            )
        else:
            # 启用几何输入：使用确定性概率（推理时不做随机dropout）
            self.geometric_input_config.update(
                {
                    "overall_prob": 1.0,  # 确定使用几何输入
                    "dropout_prob": 0.0,  # 不做dropout
                    "ray_dirs_prob": 1.0 if use_calibration else 0.0,  # 根据参数决定是否使用光线方向
                    "depth_prob": 1.0 if use_depth else 0.0,  # 根据参数决定是否使用深度
                    "cam_prob": 1.0 if use_pose else 0.0,  # 根据参数决定是否使用相机位姿
                    "sparse_depth_prob": 0.0,  # 推理时不使用稀疏深度
                    "depth_scale_norm_all_prob": 0.0 if use_depth_scale else 1.0,  # 0=提供尺度编码, 1=不提供
                    "pose_scale_norm_all_prob": 0.0 if use_pose_scale else 1.0,  # 0=提供尺度编码, 1=不提供
                }
            )

    def _restore_original_geometric_input_config(self):
        """
        恢复原始几何输入配置。
        在推理结束后调用，确保不影响后续训练时的随机行为。
        """
        if hasattr(self, "_original_geometric_config"):
            self.geometric_input_config.update(self._original_geometric_config)

    # ============================================================
    # 【重要代码段】用户友好的推理接口
    # 重要性：这是MapAnything模型对外暴露的主要推理API。相比forward()方法，
    # infer()方法提供了完整的推理流水线，包括：
    #   1. 输入验证和自动转换（如intrinsics->ray_directions, 4x4矩阵->四元数+平移）
    #   2. 几何输入配置（确定性地开/关各几何模态）
    #   3. 自动混合精度（AMP）推理
    #   4. 输出后处理（掩码、边缘检测、置信度过滤、多视图一致性置信度等）
    # 这使得用户无需了解内部数据格式，直接提供原始输入即可获得高质量结果。
    # ============================================================
    @torch.inference_mode()  # 禁用梯度计算，减少显存占用并加速推理
    def infer(
        self,
        views: List[Dict[str, Any]],  # 视图字典列表，支持灵活的输入格式
        memory_efficient_inference: bool = True,  # 是否使用内存高效推理（默认开启）
        minibatch_size: int = None,  # 可选的固定小批量大小
        use_amp: bool = True,  # 是否使用自动混合精度（加速推理）
        amp_dtype: str = "bf16",  # 混合精度类型：bf16/fp16/fp32
        apply_mask: bool = True,  # 是否应用非模糊区域掩码
        mask_edges: bool = True,  # 是否计算和应用边缘掩码
        edge_normal_threshold: float = 5.0,  # 法线边缘检测阈值
        edge_depth_threshold: float = 0.03,  # 深度边缘检测相对阈值
        apply_confidence_mask: bool = False,  # 是否应用置信度掩码
        confidence_percentile: float = 10,  # 置信度过滤百分位阈值
        ignore_calibration_inputs: bool = False,  # 是否忽略标定输入
        ignore_depth_inputs: bool = False,  # 是否忽略深度输入
        ignore_pose_inputs: bool = False,  # 是否忽略位姿输入
        ignore_depth_scale_inputs: bool = False,  # 是否忽略深度尺度输入
        ignore_pose_scale_inputs: bool = False,  # 是否忽略位姿尺度输入
        use_multiview_confidence: bool = False,  # 是否使用多视图深度一致性置信度
        multiview_conf_depth_abs_thresh: float = 0.02,  # 多视图置信度的绝对深度阈值
        multiview_conf_depth_rel_thresh: float = 0.02,  # 多视图置信度的相对深度阈值
    ) -> List[Dict[str, torch.Tensor]]:
        """
        User-friendly inference with strict input validation and automatic conversion.

        Args:
            views: List of view dictionaries. Each dict can contain:
                Required:
                - 'img': torch.Tensor of shape (B, 3, H, W) - normalized RGB images
                - 'data_norm_type': str - normalization type used to normalize the images (must be equal to self.model.encoder.data_norm_type)

                Optional Geometric Inputs (only one of intrinsics OR ray_directions):
                - 'intrinsics': torch.Tensor of shape (B, 3, 3) - will be converted to ray directions
                - 'ray_directions': torch.Tensor of shape (B, H, W, 3) - ray directions in camera frame
                - 'depth_z': torch.Tensor of shape (B, H, W, 1) - Z depth in camera frame (intrinsics or ray_directions must be provided)
                - 'camera_poses': torch.Tensor of shape (B, 4, 4) or tuple of (quats - (B, 4), trans - (B, 3)) - can be any world frame
                - 'is_metric_scale': bool or torch.Tensor of shape (B,) - if not provided, defaults to True

                Optional Additional Info:
                - 'instance': List[str] where length of list is B - instance info for each view
                - 'idx': List[int] where length of list is B - index info for each view
                - 'true_shape': List[tuple] where length of list is B - true shape info (H, W) for each view

            memory_efficient_inference: Whether to use memory-efficient inference for dense prediction heads (trades off speed). Defaults to True.
            minibatch_size: Optional fixed minibatch size for memory-efficient inference. If provided, skips dynamic computation based on available GPU memory. Defaults to None (adaptive).
            use_amp: Whether to use automatic mixed precision for faster inference. Defaults to True.
            amp_dtype: The dtype to use for mixed precision. Defaults to "bf16" (bfloat16). Options: "fp16", "bf16", "fp32".
            apply_mask: Whether to apply the non-ambiguous mask to the output. Defaults to True.
            mask_edges: Whether to compute an edge mask based on normals and depth and apply it to the output. Defaults to True.
            edge_normal_threshold: Tolerance threshold for normals-based edge detection. Defaults to 5.0.
            edge_depth_threshold: Relative tolerance threshold for depth-based edge detection. Defaults to 0.03.
            apply_confidence_mask: Whether to apply the confidence mask to the output. Defaults to False.
            confidence_percentile: The percentile to use for the confidence threshold. Defaults to 10.
            ignore_calibration_inputs: Whether to ignore the calibration inputs (intrinsics and ray_directions). Defaults to False.
            ignore_depth_inputs: Whether to ignore the depth inputs. Defaults to False.
            ignore_pose_inputs: Whether to ignore the pose inputs. Defaults to False.
            ignore_depth_scale_inputs: Whether to ignore the depth scale inputs. Defaults to False.
            ignore_pose_scale_inputs: Whether to ignore the pose scale inputs. Defaults to False.
            use_multiview_confidence: Whether to compute multi-view depth consistency confidence instead of
                using learning-based confidence. For single-view inference, returns all ones.
                Note: This adds memory and compute overhead proportional to view count. Defaults to False.
            multiview_conf_depth_abs_thresh: Absolute depth threshold for multi-view confidence inlier matching.
                Defaults to 0.02.
            multiview_conf_depth_rel_thresh: Relative depth threshold for multi-view confidence inlier matching.
                Defaults to 0.02.

        IMPORTANT CONSTRAINTS:
        - Cannot provide both 'intrinsics' and 'ray_directions' (they represent the same information)
        - If 'depth' is provided, then 'intrinsics' or 'ray_directions' must also be provided
        - If ANY view has 'camera_poses', then view 0 (first view) MUST also have 'camera_poses'

        Returns:
            List of prediction dictionaries, one per view. Each dict contains:
                - 'img_no_norm': torch.Tensor of shape (B, H, W, 3) - denormalized rgb images
                - 'pts3d': torch.Tensor of shape (B, H, W, 3) - predicted points in world frame
                - 'pts3d_cam': torch.Tensor of shape (B, H, W, 3) - predicted points in camera frame
                - 'ray_directions': torch.Tensor of shape (B, H, W, 3) - ray directions in camera frame
                - 'intrinsics': torch.Tensor of shape (B, 3, 3) - pinhole camera intrinsics recovered from ray directions
                - 'depth_along_ray': torch.Tensor of shape (B, H, W, 1) - depth along ray in camera frame
                - 'depth_z': torch.Tensor of shape (B, H, W, 1) - Z depth in camera frame
                - 'cam_trans': torch.Tensor of shape (B, 3) - camera translation in world frame
                - 'cam_quats': torch.Tensor of shape (B, 4) - camera quaternion in world frame
                - 'camera_poses': torch.Tensor of shape (B, 4, 4) - camera pose in world frame
                - 'metric_scaling_factor': torch.Tensor of shape (B,) - applied metric scaling factor
                - 'mask': torch.Tensor of shape (B, H, W, 1) - combo of non-ambiguous mask, edge mask and confidence-based mask if used
                - 'non_ambiguous_mask': torch.Tensor of shape (B, H, W) - non-ambiguous mask
                - 'non_ambiguous_mask_logits': torch.Tensor of shape (B, H, W) - non-ambiguous mask logits
                - 'conf': torch.Tensor of shape (B, H, W) - confidence

        Raises:
            ValueError: 输入无效、缺少必需键、模态冲突或违反约束时抛出
        """
        # 确定混合精度浮点类型
        if use_amp:
            if amp_dtype == "fp16":
                amp_dtype = torch.float16  # 半精度浮点
            elif amp_dtype == "bf16":
                if torch.cuda.is_bf16_supported():
                    amp_dtype = torch.bfloat16  # BFloat16（推荐，数值稳定性更好）
                else:
                    warnings.warn(
                        "bf16 is not supported on this device. Using fp16 instead."
                    )
                    amp_dtype = torch.float16  # 设备不支持bf16时回退到fp16
            elif amp_dtype == "fp32":
                amp_dtype = torch.float32  # 全精度（最慢但最精确）
        else:
            amp_dtype = torch.float32  # 不使用AMP时使用全精度

        # 步骤1：验证输入视图的格式和约束
        validated_views = validate_input_views_for_inference(views)

        # 步骤2：将视图数据传输到模型所在的设备（GPU）
        ignore_keys = set(  # 这些键不是张量，不需要传输到GPU
            [
                "instance",
                "idx",
                "true_shape",
                "data_norm_type",
            ]
        )
        for view in validated_views:
            for name in view.keys():
                if name in ignore_keys:
                    continue
                val = view[name]
                if name == "camera_poses" and isinstance(val, tuple):
                    # camera_poses可能是元组(quats, trans)形式
                    view[name] = tuple(
                        x.to(self.device, non_blocking=True) for x in val  # non_blocking允许异步传输
                    )
                elif hasattr(val, "to"):
                    view[name] = val.to(self.device, non_blocking=True)  # 将张量传到GPU

        # 步骤3：预处理输入视图（如intrinsics->ray_directions, 4x4矩阵->四元数+平移, z_depth->depth_along_ray等转换）
        processed_views = preprocess_input_views_for_inference(validated_views)

        # 步骤4：根据参数配置几何输入的使用方式（确定性开/关）
        self._configure_geometric_input_config(
            use_calibration=not ignore_calibration_inputs,
            use_depth=not ignore_depth_inputs,
            use_pose=not ignore_pose_inputs,
            use_depth_scale=not ignore_depth_scale_inputs,
            use_pose_scale=not ignore_pose_scale_inputs,
        )

        # 步骤5：运行模型前向传播（核心推理步骤）
        with torch.autocast("cuda", enabled=bool(use_amp), dtype=amp_dtype):  # 使用自动混合精度加速
            preds = self.forward(
                processed_views,
                memory_efficient_inference=memory_efficient_inference,
                minibatch_size=minibatch_size,
            )

        # 步骤6：后处理模型输出
        # 包括：掩码应用、边缘检测、置信度过滤、多视图一致性置信度计算等
        preds = postprocess_model_outputs_for_inference(
            raw_outputs=preds,  # 模型原始输出
            input_views=processed_views,  # 预处理后的输入视图
            apply_mask=apply_mask,  # 是否应用非模糊区域掩码
            mask_edges=mask_edges,  # 是否检测并掩码边缘
            edge_normal_threshold=edge_normal_threshold,  # 法线边缘阈值
            edge_depth_threshold=edge_depth_threshold,  # 深度边缘阈值
            apply_confidence_mask=apply_confidence_mask,  # 是否应用置信度掩码
            confidence_percentile=confidence_percentile,  # 置信度百分位阈值
            use_multiview_confidence=use_multiview_confidence,  # 是否使用多视图一致性置信度
            multiview_conf_depth_abs_thresh=multiview_conf_depth_abs_thresh,  # 多视图绝对深度阈值
            multiview_conf_depth_rel_thresh=multiview_conf_depth_rel_thresh,  # 多视图相对深度阈值
        )

        # 步骤7：恢复原始几何输入配置（确保不影响后续训练）
        self._restore_original_geometric_input_config()

        return preds  # 返回后处理后的预测结果


class FishEyeMapAnything(MapAnything):
    "Modular MapAnything model class that supports input of images & optional geometric modalities (multiple reconstruction tasks)."

    def __init__(
        self,
        name: str,
        encoder_config: Dict,
        info_sharing_config: Dict,
        pred_head_config: Dict,
        geometric_input_config: Dict,
        fusion_norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(
            nn.LayerNorm, eps=1e-6
        ),
        pretrained_checkpoint_path: str = None,
        load_specific_pretrained_submodules: bool = False,
        specific_pretrained_submodules: list = None,
        torch_hub_force_reload: bool = False,
        use_register_tokens_from_encoder: bool = False,
        info_sharing_mlp_layer_str: str = "mlp",
    ):
        """
        Multi-view model containing an image encoder fused with optional geometric modalities followed by a multi-view attention transformer and respective downstream heads.
        The goal is to output scene representation.
        The multi-view attention transformer also takes as input a scale token to predict the metric scaling factor for the predicted scene representation.

        Args:
            name (str): Name of the model.
            encoder_config (Dict): Configuration for the encoder.
            info_sharing_config (Dict): Configuration for the multi-view attention transformer.
            pred_head_config (Dict): Configuration for the prediction heads.
            geometric_input_config (Dict): Configuration for the input of optional geometric modalities.
            fusion_norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Normalization layer to use after fusion (addition) of encoder and geometric modalities. (default: partial(nn.LayerNorm, eps=1e-6))
            pretrained_checkpoint_path (str): Path to pretrained checkpoint. (default: None)
            load_specific_pretrained_submodules (bool): Whether to load specific pretrained submodules. (default: False)
            specific_pretrained_submodules (list): List of specific pretrained submodules to load. Must be provided when load_specific_pretrained_submodules is True. (default: None)
            torch_hub_force_reload (bool): Whether to force reload the encoder from torch hub. (default: False)
            use_register_tokens_from_encoder (bool): Whether to use register tokens from encoder. (default: False)
            info_sharing_mlp_layer_str (str): Type of MLP layer to use in the multi-view transformer. Useful for DINO init of the multi-view transformer. Options: "mlp" or "swiglufused". (default: "mlp")
        """
        super(MapAnything, self).__init__()

        # Initialize the attributes
        self.name = name
        self.encoder_config = encoder_config
        self.info_sharing_config = info_sharing_config
        self.pred_head_config = pred_head_config
        self.geometric_input_config = geometric_input_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.load_specific_pretrained_submodules = load_specific_pretrained_submodules
        self.specific_pretrained_submodules = specific_pretrained_submodules
        self.torch_hub_force_reload = torch_hub_force_reload
        self.use_register_tokens_from_encoder = use_register_tokens_from_encoder
        self.info_sharing_mlp_layer_str = info_sharing_mlp_layer_str
        self.class_init_args = {
            "name": self.name,
            "encoder_config": self.encoder_config,
            "info_sharing_config": self.info_sharing_config,
            "pred_head_config": self.pred_head_config,
            "geometric_input_config": self.geometric_input_config,
            "pretrained_checkpoint_path": self.pretrained_checkpoint_path,
            "load_specific_pretrained_submodules": self.load_specific_pretrained_submodules,
            "specific_pretrained_submodules": self.specific_pretrained_submodules,
            "torch_hub_force_reload": self.torch_hub_force_reload,
            "use_register_tokens_from_encoder": self.use_register_tokens_from_encoder,
            "info_sharing_mlp_layer_str": self.info_sharing_mlp_layer_str,
        }

        # Get relevant parameters from the configs
        self.info_sharing_type = info_sharing_config["model_type"]
        self.info_sharing_return_type = info_sharing_config["model_return_type"]
        self.pred_head_type = pred_head_config["type"]

        # Initialize image encoder
        if self.encoder_config["uses_torch_hub"]:
            self.encoder_config["torch_hub_force_reload"] = torch_hub_force_reload
        # Create a copy of the config before deleting the key to preserve it for serialization
        encoder_config_copy = self.encoder_config.copy()
        del encoder_config_copy["uses_torch_hub"]
        self.encoder = encoder_factory(**encoder_config_copy)

        # Initialize the encoder for ray directions
        ray_dirs_encoder_config = self.geometric_input_config["ray_dirs_encoder_config"]
        ray_dirs_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        ray_dirs_encoder_config["patch_size"] = self.encoder.patch_size
        self.ray_dirs_encoder = encoder_factory(**ray_dirs_encoder_config)

        # Initialize the encoder for depth (normalized per view and values after normalization are scaled logarithmically)
        depth_encoder_config = self.geometric_input_config["depth_encoder_config"]
        depth_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        depth_encoder_config["patch_size"] = self.encoder.patch_size
        self.depth_encoder = encoder_factory(**depth_encoder_config)

        # Initialize the encoder for log scale factor of depth
        depth_scale_encoder_config = self.geometric_input_config["scale_encoder_config"]
        depth_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.depth_scale_encoder = encoder_factory(**depth_scale_encoder_config)

        # Initialize the encoder for camera rotation
        cam_rot_encoder_config = self.geometric_input_config["cam_rot_encoder_config"]
        cam_rot_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_rot_encoder = encoder_factory(**cam_rot_encoder_config)

        # Initialize the encoder for camera translation (normalized across all provided camera translations)
        cam_trans_encoder_config = self.geometric_input_config[
            "cam_trans_encoder_config"
        ]
        cam_trans_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_trans_encoder = encoder_factory(**cam_trans_encoder_config)

        # Initialize the encoder for log scale factor of camera translation
        cam_trans_scale_encoder_config = self.geometric_input_config[
            "scale_encoder_config"
        ]
        cam_trans_scale_encoder_config["enc_embed_dim"] = self.encoder.enc_embed_dim
        self.cam_trans_scale_encoder = encoder_factory(**cam_trans_scale_encoder_config)
        
        # Initialize the fusion norm layer
        self.fusion_norm_layer = fusion_norm_layer(self.encoder.enc_embed_dim)

        # Initialize the Scale Token
        # Used to scale the final scene predictions to metric scale
        # During inference extended to (B, C, T), where T is the number of tokens (i.e., 1)
        self.scale_token = nn.Parameter(torch.zeros(self.encoder.enc_embed_dim))
        torch.nn.init.trunc_normal_(self.scale_token, std=0.02)

        # Set the MLP layer config for the info sharing transformer
        if info_sharing_mlp_layer_str == "mlp":
            info_sharing_config["module_args"]["mlp_layer"] = Mlp
        elif info_sharing_mlp_layer_str == "swiglufused":
            info_sharing_config["module_args"]["mlp_layer"] = SwiGLUFFNFused
        else:
            raise ValueError(
                f"Invalid info_sharing_mlp_layer_str: {info_sharing_mlp_layer_str}. Valid options: ['mlp', 'swiglufused']"
            )

        # Initialize the info sharing module (multi-view transformer)
        self._initialize_info_sharing(info_sharing_config)

        # Initialize the prediction heads
        self._initialize_prediction_heads(pred_head_config)

        # Initialize the final adaptors
        self._initialize_adaptors(pred_head_config)

        # Load pretrained weights
        self._load_pretrained_weights()
