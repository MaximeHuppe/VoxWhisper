# %% [markdown]
# # VoxWhisper Architectural Verification Notebook
# This notebook validates the tensor shapes across all visual-to-visual
# and language-to-visual attention interfaces in the VoxWhisper pipeline.

# %%
import torch
from src.models.vox_whisper import VoxWhisper

# %%
# Define hyperparameter configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Executing checks on: {DEVICE}")

# Initialize the VoxWhisper assembly network
model = VoxWhisper(
    input_channels=1,
    text_dim=768,
    embed_dim=128,
    channels=[16, 32, 64, 128],
    num_heads=4
).to(DEVICE)

model.eval()

# %%
# Create synthetic batch tensors
print("\nInstantiating synthetic input tensors...")
batch_size = 1
num_prompts = 3 # ["background", "optic nerve", "optic chiasm"]

# T1 Structural Input: 128x128x128
dummy_t1 = torch.rand(batch_size, 1, 128, 128, 128).to(DEVICE)

# T2 Structural Input: 128x128x128 (same target grid as T1)
dummy_t2 = torch.rand(batch_size, 1, 128, 128, 128).to(DEVICE)

# Cached Text Embeddings (e.g. from PubMedBERT)
dummy_text = torch.rand(batch_size, num_prompts, 768).to(DEVICE)

print(f"-> T1 volume tensor shape:       {dummy_t1.shape}")
print(f"-> T2 volume tensor shape:       {dummy_t2.shape}")
print(f"-> Text embeddings tensor shape:  {dummy_text.shape}")

# %%
# Run the pipeline forward pass
print("\nExecuting forward pass...")
with torch.no_grad():
    predictions = model(dummy_t1, dummy_t2, dummy_text)

# %%
# Verify outputs for Deep Supervision scale consistency
print("\nVerifying multi-scale output shape consistency:")
for idx, mask in enumerate(predictions):
    scale_factor = 2 ** (2 - idx) # Generates: 4, 2, 1
    target_res = 128 // scale_factor
    print(f"-> Stage {idx+1} Output Mask shape: {list(mask.shape)} (Target spatial: {target_res}^3)")
    
    # Assert dimensions are valid
    assert mask.shape == (batch_size, num_prompts, target_res, target_res, target_res)

print("\nValidation completed successfully. All structural interfaces align.")
# %%
