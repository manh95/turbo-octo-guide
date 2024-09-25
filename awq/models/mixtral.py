import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.utils.fused_utils import fuse_qkv
from awq.modules.fused.block import MixtralBlock
from awq.modules.fused.model import MixtralModel
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer as OldMixtralDecoderLayer,
    MixtralForCausalLM as OldMixtralForCausalLM,
    MixtralBLockSparseTop2MLP as OldMixtralBLockSparseTop2MLP,
)
from awq.modules.fused.norm import FasterTransformerRMSNorm

def _transformers_version_check():
    import transformers
    tv = transformers.__version__.split('.')
    if len(tv) == 4:
        major, minor, patch, dev = tv
    else:
        major, minor, patch = tv
    
    if int(major) == 4 and int(minor) < 37:
        raise Exception("Mixtral requires a minimum of 4.37.0.dev0: pip install git+https://github.com/huggingface/transformers.git")

class MixtralAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MixtralDecoderLayer"
    max_new_tokens_key = "max_position_embeddings"
    modules_to_not_convert = ["gate"]
    
    @staticmethod
    def fuse_layers(model: OldMixtralForCausalLM):
        fuser = MixtralFuser(model)
        fuser.fuse_transformer()
    
    @staticmethod
    def get_model_layers(model: OldMixtralForCausalLM):
        _transformers_version_check()
        return model.model.layers
    
    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def get_moe_for_scaling(module: OldMixtralDecoderLayer):
        return dict(
            scale_name="block_sparse_moe",
            scale_layer=module.block_sparse_moe,
            scale_shape=(module.block_sparse_moe.num_experts, module.block_sparse_moe.hidden_dim),
        )
    
    @staticmethod
    def move_embed(model: OldMixtralForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: OldMixtralDecoderLayer, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        # attention out
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))

        # NOTE: Scaled in awq.quantize.scale.scale_moe_experts, awq.modules.moe.ScaledMixtralSparseMoeBlock
        # Experts: Not a linear layer, special handling is introduced in awq.quantize.quantizer
        layers.append(dict(
            prev_op=module.block_sparse_moe,
            layers=module.block_sparse_moe.experts,
            inp=input_feat['block_sparse_moe'],
            module2inspect=module.block_sparse_moe,
        ))

        # scaling w2        
        expert: OldMixtralBLockSparseTop2MLP
        for i, expert in enumerate(module.block_sparse_moe.experts):
            layers.append(dict(
                prev_op=expert.w3,
                layers=[expert.w2],
                inp=input_feat[f'block_sparse_moe.experts.{i}.w2'],
            ))

        return layers


class MixtralFuser:
    def __init__(self, model: OldMixtralForCausalLM):
        self.model = model

        self.mixtral_blocks: List[Tuple[str, OldMixtralDecoderLayer]] = [
            (name, module) for name, module in self.model.named_modules()
            if 'MixtralDecoderLayer'.lower() in module.__class__.__name__.lower()
        ]
    
    def fuse_transformer(self):
        blocks = []

        module: OldMixtralDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device
            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight,
                module.input_layernorm.variance_epsilon
            )
            norm_2 = FasterTransformerRMSNorm(
                module.post_attention_layernorm.weight,
                module.post_attention_layernorm.variance_epsilon
            )
            blocks.append(MixtralBlock(
                hidden_size=self.model.config.hidden_size,
                n_heads=self.model.config.num_attention_heads,
                n_kv_heads=self.model.config.num_key_value_heads,
                qkv_layer=qkv,
                o_proj=module.self_attn.o_proj,
                moe=module.block_sparse_moe,
                norm_1=norm_1,
                norm_2=norm_2,
                dev=device,
                max_seq_len=self.model.config.max_new_tokens,
                rope_theta=self.model.config.rope_theta
            ))
        
        self.model.model = MixtralModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            self.model.model.norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
