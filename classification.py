"""Rouler fait une boucle d'entraînement de 100 époques avec le dataset `FashionMNIST` et un modèle Lerp linéaire de hauteur 3"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)


loader = DataLoader(mnist, batch_size= 512, shuffle = True)
loader_test = DataLoader(mnist_test, batch_size= 64, shuffle = True)
        

class Lerping(nn.Module):
    
    """Prend deux `nn.Module` et fait une combinaison convexe avec paramètre `alpha` appris
    ### Arguments
    - m1 : Premier module
    - m2 : Second module
    - n : La taille de sortie des deux modules. ILS DOIVENT ÊTRE DE MÊME DIMENSIONS DE SORTIE
    
    ### Exemple de classe
    ```python
    class ModèleLerpLinéaire(nn.Module):
        # Modèle Lerp linéaire de hauteur 1
        def __init__(self):
            super().__init__()
            self.m1 = nn.Sequential(nn.Linear(784, 128), nn.LayerNorm(128), nn.SiLU())
            self.m2 = nn.Sequential(nn.Linear(784, 128), n.LayerNorm(128), nn.SiLU())
            self.lerp = Learping(self.c1, self.c2, 128)      
        def forward(self, x):
            return self.lerp(x)
    ```
    """
    
    def __init__(self, m1:nn, m2, n):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.alpha = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        w = self.alpha.sigmoid()
        return self.m1(x) * (1 - w) + self.m2(x) * w
  
  
class BlocLerpLinéaire(nn.Module):
    """ Lerp deux modules `nn.Linear`
    ### Arguments
    - n_in : Nombre de dimension d'entré
    - n_out : Nombre de dimension de sortie
    """
    def __init__(self, n_in, n_out):
        super().__init__()

        self.c1 = nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.SiLU())
        self.c2 = nn.Sequential(nn.Linear(n_in, n_out), nn.LayerNorm(n_out), nn.SiLU())
        self.lerp = Lerping(self.c1, self.c2, n_out) 
        
    def forward(self, x):
        return self.lerp(x)
         
    
class Modèle(nn.Module):
    def __init__(self):
        super().__init__()

        self.lerp1 = BlocLerpLinéaire(784, 256)
        self.lerp2 = BlocLerpLinéaire(784, 256)
        
        self.lerp3 = BlocLerpLinéaire(784, 256)
        self.lerp4 = BlocLerpLinéaire(784, 256)
        
        self.lerp5 = BlocLerpLinéaire(256, 64)
        self.lerp6 = BlocLerpLinéaire(256, 64)
        
        self.lerp_out = BlocLerpLinéaire(64, 10)


    def forward(self, x):
        x = x.flatten(start_dim=1)
        x1 = self.lerp1(x)
        x2 = self.lerp2(x)
        x3 = self.lerp3(x)
        x4 = self.lerp4(x)
        x5 = self.lerp5(x1 + x2)
        x6 = self.lerp6(x3 + x4)        
        return self.lerp_out(x5 + x6)
    

from tqdm import tqdm
import os
os.makedirs("sample", exist_ok=True)

def train(n_epoch=100):

    modèle = Modèle()
    critère = nn.CrossEntropyLoss()
    optimiseur = torch.optim.AdamW(modèle.parameters(), lr=5e-4)

    def train_step(batch):
        images, labels = batch

        logits = modèle(images)

        perte = critère(logits, labels)

        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()

        pred = torch.argmax(logits, dim=1)
        bonnes_pred = (pred == labels).sum().item()

        return perte.item(), bonnes_pred, labels.size(0)

    bar_epoch = tqdm(range(n_epoch), desc='Entraînement', leave=True)
    
    for epoch in bar_epoch:
        batches = tqdm(loader, desc=f"Époque {epoch + 1}", leave=False)

        perte_total = 0
        bonnes_total = 0
        total = 0

        for batch in batches:
            perte_batch, bonnes_pred, n = train_step(batch)

            perte_total += perte_batch
            bonnes_total += bonnes_pred
            total += n

            batches.set_postfix({
                "Perte lot": f"{perte_batch:.6f}",
                "Acc lot": f"{bonnes_pred/n:.4f}"
            })

        perte_train = perte_total/len(loader)
        acc_train = bonnes_total/total
        
        bar_epoch.set_postfix({
            "Perte époque": f"{perte_train:.6f}",
            "Acc époque": f"{acc_train:.4f}"
        })
        
        with torch.no_grad():
            test = tqdm(loader_test, desc=f"Val {epoch + 1}", leave= False)

            perte_total = 0
            bonnes_total = 0
            total = 0

            for batch in test:
                images, labels = batch

                logits = modèle(images)
                perte = critère(logits, labels)

                perte_total += perte.item()

                pred = torch.argmax(logits, dim=1)
                bonnes = (pred == labels).sum().item()

                bonnes_total += bonnes
                total += labels.size(0)

                test.set_postfix({
                    "perte val": f"{perte.item():.6f}",
                    "acc val": f"{bonnes / labels.size(0):.4f}"
                })

            perte_val = perte_total / len(loader_test)
            acc_val = bonnes_total / total
            
            bar_epoch.set_postfix({
                "Perte val": f"{perte_val:.6f}",
                "Acc val": f"{acc_val:.4f}"
            })

            tqdm.write(
            f"Époque {epoch + 1:>3}/{n_epoch}"
            f"  │  perte train: {perte_train:.4f}  acc train: {acc_train:.4f}"
            f"  │  perte val:   {perte_val:.4f}  acc val:   {acc_val:.4f}"
        )   
            fig, axes = plt.subplots(2, 5, figsize = (12,8))

            batch, tags = next(iter(loader_test))

            for i in range(10):
                img = batch[i]

                logits = modèle(img.unsqueeze(0)).squeeze(0)
                pred = torch.argmax(logits).item()

                img_np = img.permute(1, 2, 0).detach()

                axes[i//5, i%5].set_title(f"Pred : {str(pred)}   Vrai: {tags[i].squeeze().item()}")
                axes[i//5, i%5].imshow(img_np, cmap='gray')
                axes[i//5, i%5].axis('off')

            plt.savefig(f"./sample/sample_{epoch + 1}")
            plt.close()
                
    torch.save(modèle.state_dict(), "modèle_lerpLinéaire_FashionMNIST.pth")
    

train()      
        
        
        
