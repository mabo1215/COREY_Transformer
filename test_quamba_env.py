#!/usr/bin/env python3
"""
Quick validation: load mamba-370m, test inference, confirm GPU/library stacks.
"""
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("=" * 60)
print("QUAMBA-PY310 Environment Validation")
print("=" * 60)

print(f"\n[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

print("\n[INFO] Loading mamba-370m checkpoint...")
try:
    model_id = "state-spaces/mamba-370m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map="cuda"
    )
    print(f"[SUCCESS] Model loaded: {model.config.model_type}")
    print(f"[INFO] Model architecture: {model.config.hidden_size}d x {model.config.num_hidden_layers}L")
    print(f"[INFO] Vocab size: {len(tokenizer)}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

print("\n[INFO] Running inference test...")
try:
    prompt = "Hello, this is a test of the Mamba model for potential quantization."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"[INFO] Input prompt length: {inputs.input_ids.shape[1]} tokens")
    
    with torch.no_grad():
        outputs = model(**inputs, max_new_tokens=16)
    
    output_shape = outputs.logits.shape if hasattr(outputs, 'logits') else outputs.last_hidden_state.shape
    print(f"[SUCCESS] Inference completed. Output shape: {output_shape}")
except Exception as e:
    print(f"[ERROR] Inference failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] QUAMBA-PY310 environment is fully functional!")
print("[INFO] Ready for quantization/evaluation experiments")
print("=" * 60 + "\n")
