import torch
import wandb

def run_fingerprint(model: torch.nn.Module) -> wandb.Table:
    """
    Computes weight statistics (mean, std, sparsity, frobenius norm) 
    per layer and returns a wandb.Table.
    """
    columns = ["Layer Name", "Shape", "Mean", "Std", "Abs Mean", "Max", "Min", "Sparsity", "Frobenius"]
    data = []
    
    for name, param in model.named_parameters():
        if ".layers." not in name or param.ndim < 2:
            continue
            
        p_data = param.data.float()
        
        row = [
            name,
            str(list(param.shape)),
            p_data.mean().item(),
            p_data.std().item(),
            p_data.abs().mean().item(),
            p_data.max().item(),
            p_data.min().item(),
            (p_data.abs() < 1e-6).float().mean().item(),
            p_data.norm().item()
        ]
        data.append(row)
        
    return wandb.Table(columns=columns, data=data)
