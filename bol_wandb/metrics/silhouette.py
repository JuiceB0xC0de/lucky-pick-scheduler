from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy import stats as scipy_stats

from ..config import RELATED_PAIRS, UNRELATED_PAIRS
from .common import resolve_layer_count


def run_silhouette(model: torch.nn.Module, tokenizer) -> wandb.Table:
    device = next(model.parameters()).device
    model.eval()
    num_layers = resolve_layer_count(model)

    def get_layers(word):
        inp = tokenizer(word, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        return [h[0, -1, :].cpu().float().numpy() for h in out.hidden_states]

    all_words = list({w for pair in RELATED_PAIRS + UNRELATED_PAIRS for w in pair})
    cache = {w: get_layers(w) for w in all_words}

    columns = ["Layer", "Related Avg", "Unrelated Avg", "Separation", "P-Value", "Significant"]
    data = []

    observed_layers = min(num_layers + 1, len(cache[all_words[0]]))
    for layer in range(observed_layers):
        rel_sims = [
            F.cosine_similarity(
                torch.tensor(cache[a][layer]).unsqueeze(0),
                torch.tensor(cache[b][layer]).unsqueeze(0),
            ).item()
            for a, b in RELATED_PAIRS
        ]
        unrel_sims = [
            F.cosine_similarity(
                torch.tensor(cache[a][layer]).unsqueeze(0),
                torch.tensor(cache[b][layer]).unsqueeze(0),
            ).item()
            for a, b in UNRELATED_PAIRS
        ]

        ra = np.mean(rel_sims)
        ua = np.mean(unrel_sims)
        sep = ra - ua

        if len(rel_sims) >= 3 and len(unrel_sims) >= 3:
            try:
                _, pval = scipy_stats.mannwhitneyu(rel_sims, unrel_sims, alternative="greater")
            except ValueError:
                pval = 1.0
        else:
            pval = 1.0

        data.append([layer, ra, ua, sep, pval, bool(pval < 0.05)])

    return wandb.Table(columns=columns, data=data)
