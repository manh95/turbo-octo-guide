from typing import Union
from .base import BaseAWQForCausalLM
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel, BertLayer

class BertAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "BertLayer"

    @staticmethod
    def fuse_layers(awq_model):
        # TODO: QKV can be fused
        pass

    @staticmethod
    def get_model_layers(model: Union[BertForMaskedLM, BertModel]):
        return model.bert.encoder.layer
    
    @staticmethod
    def get_act_for_scaling(module: BertLayer):
        return dict(
            is_scalable=True,
            scale_name="intermediate.intermediate_act_fn",
            scale_layer=module.intermediate.intermediate_act_fn,
            scale_shape=module.intermediate.dense
        )
    
    @staticmethod
    def move_embed(model: Union[BertForMaskedLM, BertModel], device: str):
        model.embeddings.word_embeddings = model.embeddings.word_embeddings.to(device)
        model.embeddings.position_embeddings = model.embeddings.position_embeddings.to(device)
        model.embeddings.token_type_embeddings = model.embeddings.token_type_embeddings.to(device)
    
    @staticmethod
    def get_layers_for_scaling(module: BertLayer, input_feat, module_kwargs):
        layers = []
        
        # module.attention
        # TODO: Handle NoOp. No previous LayerNorm/Linear in module.attention like in other models.
        layers.append(dict(
            prev_op=None, 
            layers=[module.attention.self.query,
                    module.attention.self.key, module.attention.self.value],
            inp=input_feat['attention.self.query'],
            module2inspect=module.attention, kwargs=module_kwargs,
        ))

        # attention out
        layers.append(dict(
            prev_op=module.attention.self.value,
            layers=[module.attention.output.dense],
            inp=input_feat['attention.self.value'],
        ))

        # linear 1
        layers.append(dict(
            prev_op=module.attention.output.dropout,
            layers=[module.intermediate.dense],
            inp=input_feat['attention.output.dropout'],
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.intermediate.intermediate_act_fn,
            layers=[module.output.dense],
            inp=input_feat['intermediate.intermediate_act_fn'],
        ))

        return layers