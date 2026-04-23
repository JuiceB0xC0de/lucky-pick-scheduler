import torch
import math
import wandb

from ..config import DEFAULT_EVAL_TEXTS

def run_layer_sweep(model: torch.nn.Module, tokenizer, eval_texts=None) -> wandb.Table:
    """
    Ablates layers one by one and computes the perplexity/loss damage.
    Returns a wandb.Table for bar chart visualization.
    """
    device = next(model.parameters()).device
    model.eval()
    
    if hasattr(model.config, 'num_hidden_layers'):
        N = model.config.num_hidden_layers
    else:
        N = len([n for n, _ in model.named_parameters() if 'layers.' in n and 'weight' in n]) // 4
        
    texts = eval_texts if eval_texts else DEFAULT_EVAL_TEXTS
    tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    labels = tokenized["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
        
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    labels = labels.to(device)

    def get_batched_loss():
        with torch.no_grad():
            outputs = model(**tokenized, labels=labels)
        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return float('nan')
        return loss.item()

    # Save original weights to restore them
    original_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue # Try to avoid saving adam states if training
        original_weights[name] = param.data.clone()

    baseline_loss = get_batched_loss()
    
    columns = ["Layer", "Loss", "Damage", "Impact"]
    data = []

    for i in range(N):
        zeroed_params = []
        try:
            for name, param in model.named_parameters():
                if f".layers.{i}." in name and param.ndim >= 2:
                    param.data.zero_()
                    zeroed_params.append(name)

            if device.type == "cuda":
                torch.cuda.synchronize()

            loss = get_batched_loss()
            
            if math.isnan(loss) or math.isinf(loss):
                loss_val = 999.0
                damage = 999.0
                tag = "CRITICAL"
            else:
                loss_val = loss
                damage = loss - baseline_loss
                tag = "CRITICAL" if damage > 1.0 else "MODERATE" if damage > 0.3 else "REMOVABLE"

            data.append([f"L{i}", loss_val, damage, tag])

        finally:
            for name in zeroed_params:
                param = model.get_parameter(name)
                param.data.copy_(original_weights[name])
                
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
    return wandb.Table(columns=columns, data=data)

def run_component_ablation(model: torch.nn.Module, tokenizer, eval_texts=None) -> wandb.Table:
    """
    Ablates component types (e.g. all q_proj) across all layers and computes damage.
    Returns a wandb.Table for bar chart visualization.
    """
    device = next(model.parameters()).device
    model.eval()
    
    COMPONENTS = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    
    texts = eval_texts if eval_texts else DEFAULT_EVAL_TEXTS
    tokenized = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    labels = tokenized["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
        
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    labels = labels.to(device)

    def get_loss():
        with torch.no_grad():
            outputs = model(**tokenized, labels=labels)
        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss):
            return float('nan')
        return loss.item()

    original_weights = {}
    for name, param in model.named_parameters():
        original_weights[name] = param.data.clone()

    baseline_loss = get_loss()
    
    columns = ["Component", "Zeroed_Tensors", "Loss", "Damage", "Impact"]
    data = []

    for comp in COMPONENTS:
        zeroed_params = []
        try:
            for name, param in model.named_parameters():
                if comp in name and param.ndim >= 2:
                    param.data.zero_()
                    zeroed_params.append(name)

            if device.type == "cuda":
                torch.cuda.synchronize()

            loss = get_loss()
            
            if math.isnan(loss) or math.isinf(loss):
                loss_val = 999.0
                damage = 999.0
                tag = "CRITICAL"
            else:
                loss_val = loss
                damage = loss - baseline_loss
                tag = "CRITICAL" if damage > 1.0 else "MODERATE" if damage > 0.3 else "REMOVABLE"

            data.append([comp, len(zeroed_params), loss_val, damage, tag])

        finally:
            for name in zeroed_params:
                param = model.get_parameter(name)
                param.data.copy_(original_weights[name])
                
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
    return wandb.Table(columns=columns, data=data)
