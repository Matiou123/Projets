from transformer import decode, CharacterTransformer

import yaml, torch

with open("transformer.yaml", "r") as f:
    config = yaml.safe_load(f)
modèle = CharacterTransformer(**config)

checkpoint = torch.load("transformer.pth", map_location="cuda")
modèle.load_state_dict(checkpoint)
modèle.to("cuda")

contexte = torch.zeros((1,1), dtype=torch.long, device="cuda")
print(decode(modèle.generate(contexte, max_new_tokens= 2000)[0].tolist()))