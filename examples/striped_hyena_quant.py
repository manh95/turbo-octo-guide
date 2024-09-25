from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'togethercomputer/StripedHyena-Nous-7B'
quant_path = 'striped-hyna-nous-awq'
modules_to_not_convert = ["out_filter_dense"]
quant_config = { 
    "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM",
    "modules_to_not_convert": modules_to_not_convert
}

# Load model
# NOTE: pass safetensors=True to load safetensors
model = AutoAWQForCausalLM.from_pretrained(
    model_path, safetensors=True, **{"low_cpu_mem_usage": True, "use_cache": False, "use_flash_rmsnorm": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    modules_to_not_convert=modules_to_not_convert
)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')