import re
import torch
import torch.nn as nn
from functools import partial
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm

ALLOWED_NORMS = (nn.LayerNorm, LlamaRMSNorm)
ALLOWED_MODULES = (nn.Linear,) + ALLOWED_NORMS

def split_layer_name(name):
    pattern = r"(.*[^0-9])\.(\d+)\.(.+)"
    match = re.match(pattern, name)

    if match is not None:
        # prefix, index, suffix
        prefix, index, suffix = match.groups()
        suffix_parent, suffix_module = '.'.join(suffix.split('.')[:-1]), suffix.split('.')[-1]
        return prefix, index, suffix_parent, suffix_module
    else:
        return None, None, None, None

class OrderedModelDefinition(nn.Module):
    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model
        self.hooks = []
        self.ordered_modules = {}
        self._register_hooks()
    
    def forward(self):
        """Execute the model to collect ordered modules."""
        self.ordered_modules.clear()
        x = torch.randint(0, 1, (1, 1))
        self.model(x)
        self._remove_hooks()

    def collect_modules(self, model, input, output, module, name):
        """Collect allowed modules."""
        if isinstance(module, ALLOWED_MODULES):
            self.ordered_modules[name] = module
    
    def _register_hooks(self):
        """Register hooks to collect modules."""
        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(partial(self.collect_modules, module=module, name=name))
            self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class LayerExtraction:
    def __init__(self, model) -> None:
        self.layers = []
        self.model = model
        self.model_definition = OrderedModelDefinition(model)

        # define indices
        self.qkv_last_index = 0
        self.post_qkv_index = 0
        self.mlp_last_index = 0
        self.last_module_suffix_parent = None
        # (name, module)
        self.last_norm = None
        # [(name, module), ...]
        self.last_linears = []
        self.accumulated_modules = []
    
    def run(self):
        """Execute the layer extraction."""
        self.model_definition.forward()

        for name, module in self.model_definition.ordered_modules.items():
            self._process_module(name, module)
    
    def _process_module(self, name, module):
        """Process each module to extract layers."""
        prefix, index, suffix_parent, suffix_module = split_layer_name(name)

        # skips lm head
        if index is None:
            return

        self._update_layers(module, name, index, suffix_parent)
        self._append_qkv_layers(index)
        self._append_post_qkv(index, name, module)
        self._append_mlp_layers(index, name, module)
        self._append_post_mlp(index, name, module)
    
    def _update_layers(self, module, name, index, suffix_parent):
        """Update layer based on the type of module."""
        # do not allow cross-module
        if suffix_parent != self.last_module_suffix_parent:
            self.last_linears = []

        if isinstance(module, nn.Linear):
            self.last_linears.append((name, module))
            self._update_last_linears(index, name, module)
        
        if self._is_module_allowed(module, ALLOWED_NORMS):
            self.last_norm = (name, module)
        
        self.accumulated_modules.append((name, module))
        self.last_module_suffix_parent = suffix_parent
    
    def _update_last_linears(self, index, name, module):
        """Update last_linears if a new module is encountered."""
        # if new module, only save last_linears from new module
        is_new_module = any(index > split_layer_name(linear[0])[1] for linear in self.last_linears)
        if is_new_module:
            self.last_linears = self.last_linears[-1:]
    
    def _append_qkv_layers(self, index):
        # norm + qkv
        # if we found norm and we have 3 linears available
        if self.last_norm is not None and \
            self._is_module_allowed(self.last_norm[1], ALLOWED_NORMS) and len(self.last_linears) == 3:

            # are the last 3 modules a linear module
            if self.qkv_last_index != index and all(
                    isinstance(m, nn.Linear) for n,m in self.accumulated_modules[-3:]):
                self.layers.append({"prev_op": self.last_norm, "layers": self.last_linears})
                self.last_linears = []
                self.last_norm = None
                self.qkv_last_index = index
    
    def _append_post_qkv(self, index, name, module):
        # v_proj + o_proj
        if self.qkv_last_index == index and len(self.layers) > 0:
            prev_op = self.layers[-1]["layers"][-1]

            # is the last module [-2] the same as prev_op module
            if self.post_qkv_index != index and self.accumulated_modules[-2][1] is prev_op[1]:
                self.layers.append({"prev_op": prev_op, "layers": [(name, module)]})
                self.post_qkv_index = index
    
    def _append_mlp_layers(self, index, name, module):
        # norm + gate + up_proj
        if self.post_qkv_index == index:

            # is the second last module [-3] a norm layer and is the current module a linear
            if self._is_module_allowed(self.accumulated_modules[-3][1], ALLOWED_NORMS) and isinstance(module, nn.Linear):
                self.layers.append(
                    {"prev_op": self.accumulated_modules[-3], 
                     "layers": [self.accumulated_modules[-2], (name, module)]})
                self.mlp_last_index = index
    
    def _append_post_mlp(self, index, name, module):
        # up_proj + down_proj
        if self.mlp_last_index == index:
            prev_op = self.layers[-1]["layers"][-1]

            # if the last module [-2] is the same as the prev_op
            if self.accumulated_modules[-2][1] is prev_op[1]:
                self.layers.append({"prev_op": prev_op, "layers": [(name, module)]})

    @staticmethod
    def _is_module_allowed(module, allowed_modules):
        return any(isinstance(module, m) for m in allowed_modules)

def test_llama(extraction: LayerExtraction):
    llama_prev_op = [
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.mlp.up_proj",
        "model.layers.1.input_layernorm",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.post_attention_layernorm",
        "model.layers.1.mlp.up_proj"
    ]
    llama_layers = [
        ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.self_attn.v_proj"], 
        ["model.layers.0.self_attn.o_proj"], 
        ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"], 
        ["model.layers.0.mlp.down_proj"], 
        ["model.layers.1.self_attn.q_proj", "model.layers.1.self_attn.k_proj", "model.layers.1.self_attn.v_proj"], 
        ["model.layers.1.self_attn.o_proj"], 
        ["model.layers.1.mlp.gate_proj", "model.layers.1.mlp.up_proj"], 
        ["model.layers.1.mlp.down_proj"]
    ]

    prev_ops = []
    layers = []

    for layer in extraction.layers:
        prev_ops.append(layer["prev_op"][0])
        layers.append([name for name, module in layer["layers"]])

    assert prev_ops == llama_prev_op, "Previous operations not the same as llama"
    assert layers == llama_layers, "Previous layers are not the same as llama"

model = AutoModelForCausalLM.from_pretrained(
    "JackFram/llama-68m",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

extraction = LayerExtraction(model)
extraction.run()
test_llama(extraction)