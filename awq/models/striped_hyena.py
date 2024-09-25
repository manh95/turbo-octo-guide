from .base import BaseAWQForCausalLM

class StripedHyenaAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "blocks"
    max_new_tokens_key = "max_seqlen"

    @staticmethod
    def get_model_layers(model):
        try:
            import flash_attn
        except ImportError:
            raise ImportError("Flash Attention is required for Striped Hyena: pip install flash-attn")
        
        return model.backbone.blocks
    
    @staticmethod
    def get_act_for_scaling(module):
        return dict(
            is_scalable=False
        )
    
    @staticmethod
    def move_embed(model, device: str):
        model.backbone.embedding_layer = model.backbone.embedding_layer.to(device)
        model.backbone.unembed = model.backbone.unembed.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        if module.__class__.__name__ == "ParallelGatedConvBlock":
            # NOTE: out_filter_dense has prev_op of conv1d
            layers.append(dict(
                prev_op=module.pre_norm,
                layers=[module.projections],
                inp=input_feat['projections'],
                module2inspect=module.projections,
                kwargs=module_kwargs,
            ))
        elif module.__class__.__name__ == "AttentionBlock":
            layers.append(dict(
                prev_op=module.pre_norm,
                layers=[module.inner_mha_cls.Wqkv],
                inp=input_feat['inner_mha_cls.Wqkv'],
                module2inspect=module.inner_mha_cls,
                kwargs=module_kwargs,
            ))
            layers.append(dict(
                prev_op=module.inner_mha_cls.Wqkv,
                layers=[module.inner_mha_cls.out_proj],
                inp=input_feat['inner_mha_cls.out_proj'],
            ))
        
        layers.append(dict(
            prev_op=module.post_norm,
            layers=[module.mlp.l1, module.mlp.l2],
            inp=input_feat[f'mlp.l1'],
            module2inspect=module.mlp,
            kwargs=module_kwargs,
        ))
        layers.append(dict(
            prev_op=module.mlp.l2,
            layers=[module.mlp.l3],
            inp=input_feat[f'mlp.l3'],
        ))

        return layers