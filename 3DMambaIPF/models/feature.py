import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Linear, ModuleList
import pytorch3d.ops
from .utils import *
from models.dynamic_edge_conv import DynamicEdgeConv
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn.inits import reset


from mamba_ssm.modules.mamba_simple import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from functools import partial
from .block import Block
import math


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device=None,
        dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        hidden_states = hidden_states
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

def get_knn_idx(x, y, k, offset=0):
    """
    Args:
        x: (B, N, d)
        y: (B, M, d)
    Returns:
        (B, N, k)
    """
    _, knn_idx, _ = pytorch3d.ops.knn_points(y, x, K=k+offset)
    return knn_idx[:, :, offset:]

class FeatureExtraction(Module):
    def __init__(self, k=32, input_dim=0, z_dim=0, embedding_dim=512, output_dim=3):
        super(FeatureExtraction, self).__init__()
        self.k = k
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        self.conv1 = DynamicEdgeConv(3, 16)
        self.conv2 = DynamicEdgeConv(16, 48)
        self.conv3 = DynamicEdgeConv(48, 144)
        self.conv4 = DynamicEdgeConv(16+48+144, self.embedding_dim)

        self.mamba_1 = MixerModel(d_model=16,
                                 n_layer=6,
                                 rms_norm=False,
                                 drop_out_in_block=0.,
                                 drop_path=0.)
        self.mamba_2 = MixerModel(d_model=48,
                                 n_layer=6,
                                 rms_norm=False,
                                 drop_out_in_block=0.,
                                 drop_path=0.)
        self.mamba_3 = MixerModel(d_model=144,
                                 n_layer=6,
                                 rms_norm=False,
                                 drop_out_in_block=0.,
                                 drop_path=0.)
        self.mamba_4 = MixerModel(d_model=512,
                                 n_layer=1,
                                 rms_norm=False,
                                 drop_out_in_block=0.,
                                 drop_path=0.)
        self.mamba_5 = MixerModel(d_model=256,
                                 n_layer=1,
                                 rms_norm=False,
                                 drop_out_in_block=0.,
                                 drop_path=0.)
        self.mamba_6 = MixerModel(d_model=128,
                                 n_layer=1,
                                 rms_norm=False,
                                 drop_out_in_block=0.,
                                 drop_path=0.)
        # self.mamba_7 = MixerModel(d_model=3,
        #                          n_layer=1,
        #                          rms_norm=False,
        #                          drop_out_in_block=0.,
        #                          drop_path=0.)

        self.linear1 = nn.Linear(self.embedding_dim, 256, bias=False)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, self.output_dim)

        if self.z_dim > 0:
            self.linear_proj = nn.Linear(512, self.z_dim)
            self.dropout_proj = nn.Dropout(0.1)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.conv1)
        reset(self.conv2)
        reset(self.conv3)
        reset(self.conv4)
        reset(self.linear1)
        reset(self.linear2)
        reset(self.linear3)

    @property
    def out_channels(self):
        return self.convs[-1].out_channels

    def get_edge_index(self, x):
        cols = get_knn_idx(x, x, self.k+1).view(self.batch_size, self.num_points, -1)
        cols = (cols + self.rows_add).view(1, -1)
        edge_index = torch.cat([cols, self.rows], dim=0)
        edge_index, _ = remove_self_loops(edge_index.long())

        return edge_index

    def forward(self, x, disp_feat):
        self.batch_size = x.size(0)
        self.num_points = x.size(1)

        x_device = x.device
        self.rows = torch.arange(0, self.num_points).unsqueeze(0).unsqueeze(2).repeat(self.batch_size, 1, self.k+1).to(x_device)
        self.rows_add = (self.num_points*torch.arange(0, self.batch_size)).unsqueeze(1).unsqueeze(2).repeat(1, self.num_points, self.k+1).to(x_device)
        self.rows = (self.rows + self.rows_add).view(1, -1)

        if disp_feat is not None:
            disp_feat = F.relu(self.linear_proj(disp_feat))
            disp_feat = self.dropout_proj(disp_feat)
            x = torch.cat([x, disp_feat], dim=-1)        
        
        edge_index = self.get_edge_index(x)
        x = x.view(self.batch_size*self.num_points, -1)
        x1 = self.conv1(x, edge_index)
        x1 = x1.view(self.batch_size, self.num_points, -1)
        x1 = self.mamba_1(x1)

        edge_index = self.get_edge_index(x1)
        x1 = x1.view(self.batch_size*self.num_points, -1)
        x2 = self.conv2(x1, edge_index)
        x2 = x2.view(self.batch_size, self.num_points, -1)
        x2 = self.mamba_2(x2)

        edge_index = self.get_edge_index(x2)
        x2 = x2.view(self.batch_size*self.num_points, -1)
        x3 = self.conv3(x2, edge_index)
        x3 = x3.view(self.batch_size, self.num_points, -1)
        x3 = self.mamba_3(x3)

        edge_index = self.get_edge_index(x3)
        x3 = x3.view(self.batch_size*self.num_points, -1)
        x_combined = torch.cat((x1, x2, x3), dim=-1)
        x_combined = x_combined.view(self.batch_size*self.num_points, -1)
        x = self.conv4(x_combined, edge_index)
        x = x.view(self.batch_size, self.num_points, -1)
        x = self.mamba_4(x)
        x = F.relu(self.mamba_5(self.linear1(x)))
        x = F.relu(self.mamba_6(self.linear2(x)))
        x = torch.tanh(self.linear3(x))

        x, x_combined = x.view(self.batch_size, self.num_points, -1), x_combined.view(self.batch_size, self.num_points, -1)

        if self.z_dim > 0:
            return x, x_combined.transpose(2, 1).contiguous()
        else:
            return x, None