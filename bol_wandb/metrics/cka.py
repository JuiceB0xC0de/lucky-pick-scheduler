import torch
import numpy as np
import wandb

from ..config import CLUSTER_WORDS

def centering_matrix(n):
    return np.eye(n) - np.ones((n, n)) / n

def linear_kernel(X):
    return X @ X.T

def hsic(K, L, H):
    n = K.shape[0]
    return np.trace(H @ K @ H @ L) / ((n - 1) ** 2)

def linear_cka(X, Y):
    n = X.shape[0]
    H = centering_matrix(n)
    K = linear_kernel(X)
    L = linear_kernel(Y)

    hsic_kl = hsic(K, L, H)
    hsic_kk = hsic(K, K, H)
    hsic_ll = hsic(L, L, H)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0
    return hsic_kl / denom

def run_cka(model: torch.nn.Module, tokenizer) -> wandb.Table:
    """
    Computes Centered Kernel Alignment (CKA) between layers to measure 
    representation similarity. Returns a wandb.Table for heatmap visualization.
    """
    device = next(model.parameters()).device
    model.eval()
    
    if hasattr(model.config, 'num_hidden_layers'):
        N = model.config.num_hidden_layers
    else:
        N = len([n for n, _ in model.named_parameters() if 'layers.' in n and 'weight' in n]) // 4

    def get_hidden(word, layer_idx):
        inputs = tokenizer(word, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]
        return hidden.mean(dim=1).squeeze().cpu().float().numpy()

    # Sample layers to avoid massive O(N^2) compute if N is large
    SAMPLE_LAYERS = list(range(0, N + 1, max(1, N // 10)))
    if N not in SAMPLE_LAYERS:
        SAMPLE_LAYERS.append(N)
    SAMPLE_LAYERS = sorted(set(SAMPLE_LAYERS))

    rep_matrices = {}
    for layer in SAMPLE_LAYERS:
        vecs = [get_hidden(w, layer) for w in CLUSTER_WORDS]
        rep_matrices[layer] = np.stack(vecs)

    columns = ["Layer_X", "Layer_Y", "CKA_Score"]
    data = []
    
    for li in SAMPLE_LAYERS:
        for lj in SAMPLE_LAYERS:
            cka_val = linear_cka(rep_matrices[li], rep_matrices[lj])
            data.append([f"L{li}", f"L{lj}", cka_val])
            
    return wandb.Table(columns=columns, data=data)
