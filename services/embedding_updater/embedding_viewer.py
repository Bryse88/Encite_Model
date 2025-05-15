import torch

# Load the embeddings
embeddings = torch.load("embeddings/hgt_embeddings.pt", map_location="cpu")


# Print a summary
print("Top-level keys:", embeddings.keys())

for key, value in embeddings.items():
    if isinstance(value, torch.Tensor):
        print(f"{key}: shape {value.shape}")
    elif isinstance(value, dict):
        print(f"{key}: nested dict with {len(value)} entries")
    else:
        print(f"{key}: {type(value)}")
