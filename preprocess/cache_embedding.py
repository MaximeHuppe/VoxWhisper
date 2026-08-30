# preprocess/cache_embeddings.py
import os
import torch
from transformers import AutoTokenizer, AutoModel

def cache_prompt_embeddings(prompt_list, output_path, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
    """
    Passes medical prompts through a frozen clinical language model,
    pools the token-level embeddings per phrase, and saves them to disk.
    """
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Tokenizing and encoding prompts: {prompt_list}")
    # We pad and truncate to convert the list of phrases into a clean tensor
    inputs = tokenizer(prompt_list, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state shape: [Num_Phrases, Seq_Len, 768]
        token_embeddings = outputs.last_hidden_state
        
        # MEAN POOLING: Average along the Seq_Len dimension (dim=1)
        # This reduces the shape of each phrase to a single 768-dimensional vector
        # Shape transition: [Num_Phrases, Seq_Len, 768] -> [Num_Phrases, 768]
        embeddings = token_embeddings.mean(dim=1)

    # Save to disk (we keep it as a 2D tensor [Num_Phrases, 768])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"Cached text embeddings saved to: {output_path} (Final Shape: {embeddings.shape})")

if __name__ == "__main__":
    clinical_prompts = ["background", "optic nerve", "optic chiasm"]
    target_cache_file = "cache/prompts_cn2.pt"
    
    cache_prompt_embeddings(clinical_prompts, target_cache_file)