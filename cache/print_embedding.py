# inspect_embeddings.py
import os
import torch
from src.utils.config import load_config

def inspect_prompt_embeddings(cache_path="cache/prompts_cn2.pt"):
    # 1. Verify file existence
    if not os.path.exists(cache_path):
        print(f"Error: Cache file '{cache_path}' not found.")
        print("Please run 'python preprocess/cache_embeddings.py' first to generate it.")
        return

    # 2. Load the PyTorch tensor
    # Squeeze out any unnecessary dimensions to ensure we have a clean 2D tensor [Num_Prompts, 768]
    embeddings = torch.load(cache_path, map_location=torch.device('cpu')).squeeze()

    # The original prompts used in cache_embeddings.py (ordered)
    cfg = load_config()
    prompts = cfg["data"]["prompts"]
    prompt_labels = prompts

    # 3. Dynamic adjustment if the prompt list length differs from the tensor dimension
    num_embeddings = embeddings.shape[0] if len(embeddings.shape) > 1 else 1
    if num_embeddings != len(prompt_labels):
        prompt_labels = [f"Class {i}" for i in range(num_embeddings)]

    # 4. Print formatted metadata
    print("==================================================")
    print("      VOXWHISPER PROMPT EMBEDDINGS INSPECTOR      ")
    print("==================================================")
    print(f"File Path:       {cache_path}")
    print(f"Tensor Shape:    {list(embeddings.shape)} (Queries x Embed_Dim)")
    print(f"Data Type:       {embeddings.dtype}")
    print(f"Compute Device:  {embeddings.device}")
    print("==================================================\n")

    # 5. Print a clean, readable summary of each embedding vector
    if len(embeddings.shape) == 1:
        # Handle the case where there is only 1 single prompt in the file
        print_vector_summary(0, prompt_labels[0], embeddings)
    else:
        # Loop through each prompt class
        for idx, label in enumerate(prompt_labels):
            vector = embeddings[idx]
            print_vector_summary(idx, label, vector)

def print_vector_summary(index, label, vector):
    mean_val = vector.mean().item()
    std_val = vector.std().item()
    min_val = vector.min().item()
    max_val = vector.max().item()
    
    # Extract first and last 5 elements as a list for clean terminal preview
    first_five = [round(x, 4) for x in vector[:5].tolist()]
    last_five = [round(x, 4) for x in vector[-5:].tolist()]

    print(f"Prompt {index}: '{label}'")
    print(f"  ├─ Shape:    {list(vector.shape)}")
    print(f"  ├─ Stats:    Mean: {mean_val:.4f} | Std: {std_val:.4f} | Min: {min_val:.4f} | Max: {max_val:.4f}")
    print(f"  ├─ Preview (First 5 dims): {first_five}")
    print(f"  └─ Preview (Last 5 dims):  {last_five}\n")

if __name__ == "__main__":
    inspect_prompt_embeddings()