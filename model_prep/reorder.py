import sys
import os
import torch 
from transformers import AutoModelForCausalLM


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def get_placement(activation_sums, layer):
    a = activation_sums[layer]
    perm = sorted(range(len(a)), key=lambda i: a[i], reverse=True)

    # convert to placement list, 1-based
    placement = [0] * len(a)
    for new_pos, old_idx in enumerate(perm, start=1):
        placement[old_idx] = new_pos
    return placement


def get_projection_matrix(order):
    dim = len(order)
    P = torch.zeros((dim, dim), device=DEVICE, dtype=torch.bfloat16)
    for old_idx, new_pos in enumerate(order):
        P[new_pos - 1, old_idx] = 1
    return P


def reorder_layer(layer, P):
    # reorder gate_proj
    Wg = layer.mlp.gate_proj.weight.data
    Wu = layer.mlp.up_proj.weight.data
    Wd = layer.mlp.down_proj.weight.data

    Wg_new = P @ Wg
    Wu_new = P @ Wu
    Wd_new = Wd @ P.T

    return Wg_new, Wu_new, Wd_new


if __name__ == "__main__":
    save_path = f"{PROJECT_ROOT}/models/qwen_model_reordered_mlp_100k"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    activation_sums = torch.load(f"{PROJECT_ROOT}/mlp_activation_sums_100k.pt", map_location=DEVICE)['activation_sums']

    num_layers = len(model.model.layers)
    for l in range(num_layers):
        print(f"Processing layer {l}...")
        placement = get_placement(activation_sums, l)
        P = get_projection_matrix(placement)
        
        Wg_new, Wu_new, Wd_new = reorder_layer(model.model.layers[l], P)
        model.model.layers[l].mlp.gate_proj.weight.data = Wg_new
        model.model.layers[l].mlp.up_proj.weight.data = Wu_new
        model.model.layers[l].mlp.down_proj.weight.data = Wd_new


    print(f"Saving reordered model to: {save_path}")
    model.save_pretrained(save_path)

    print("Reordered model saved successfully.")

