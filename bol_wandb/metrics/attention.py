import torch
import numpy as np
import math
import wandb

from ..config import RELATED_PAIRS, UNRELATED_PAIRS

def attention_entropy(attn_dist):
    p = attn_dist.clamp(min=1e-10)
    return -(p * p.log()).sum().item()

def max_entropy(seq_len):
    return math.log(seq_len) if seq_len > 0 else 0

def run_attention_map(model: torch.nn.Module, tokenizer) -> wandb.Table:
    """
    Computes attention entropy across layers. Low entropy = focused, 
    High entropy = diffuse/uniform. Returns wandb.Table.
    """
    device = next(model.parameters()).device
    model.eval()
    
    if hasattr(model.config, 'num_hidden_layers'):
        N = model.config.num_hidden_layers
    else:
        N = len([n for n, _ in model.named_parameters() if 'layers.' in n and 'weight' in n]) // 4
        
    def get_attention(word):
        inp = tokenizer(word, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_attentions=True)
        
        if not out.attentions:
            return None
            
        attn_per_layer = []
        for layer_attn in out.attentions:
            last_tok_attn = layer_attn[0, :, -1, :].cpu().float()
            attn_per_layer.append(last_tok_attn)
        return attn_per_layer

    all_words = list({w for pair in RELATED_PAIRS + UNRELATED_PAIRS for w in pair})
    attn_cache = {}
    
    # Try one to see if attentions are returned
    test_attn = get_attention(all_words[0])
    if test_attn is None:
        print("[BoL W&B] Warning: Model does not output attentions. Skipping attention map.")
        return wandb.Table(columns=["Layer", "Mean_Entropy", "Normalized_Entropy", "Interpretation"], data=[])
        
    for w in all_words:
        attn_cache[w] = get_attention(w)

    columns = ["Layer", "Mean_Entropy", "Normalized_Entropy", "Interpretation"]
    data = []

    for layer in range(N):
        layer_entropies = []
        for word, attn_layers in attn_cache.items():
            attn = attn_layers[layer] 
            seq_len = attn.shape[1]
            for head in range(attn.shape[0]):
                h = attention_entropy(attn[head])
                layer_entropies.append(h)

        mean_h = np.mean(layer_entropies)
        max_possible = max_entropy(seq_len)
        norm_h = mean_h / max_possible if max_possible > 0 else 0
        
        interp = "focused" if norm_h < 0.5 else "moderate" if norm_h < 0.8 else "diffuse"
        
        data.append([layer, mean_h, norm_h, interp])
        
    return wandb.Table(columns=columns, data=data)
