import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
FashionMNIST_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
FashionMNIST_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)


loader = DataLoader(FashionMNIST_train, batch_size= 512, shuffle = True)
loader_test = DataLoader(FashionMNIST_test, batch_size= 64, shuffle = True)
        

    
class Conv(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__() 
        assert c_out % 4 == 0
        c = c_out // 4
        
        self.conv1 = nn.Conv2d(c_in, c * 2, 3, 1, 1)   
        self.conv2 = nn.Conv2d(c_in, c, 5, 1, 2)   
        self.conv3 = nn.Conv2d(c_in, c, 7, 1, 3)   
          
    
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv1(x)
        conv3 = self.conv1(x)
        
        return torch.cat([conv1, conv2, conv3], dim = 1)
    
class BlocConv(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__()
        
        self.conv = nn.Sequential(nn.GroupNorm(32, c_in), Conv(c_in, c_out), nn.SiLU(),
                                  nn.GroupNorm(32 ,c_out), Conv(c_out, c_out), nn.SiLU())
        
        self.proj = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 3, 1, 1)
    
    def forward(self, x):
        x_copie = self.proj(x)
        
        x = self.conv(x)
        return x + x_copie
        
    
class ConvDown(nn.Module):
    
    def __init__(self, c_in, c_out):
        super().__init__() 
        
        self.conv = nn.Sequential(nn.GroupNorm(32 ,c_in), nn.Conv2d(c_in, c_out, 3, 2, 1), nn.SiLU()  )
    
    def forward(self, x):
        return self.conv(x)

class Modèle(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, 1 , 1), BlocConv(32, 32), BlocConv(32, 64),  ConvDown(64, 64) , BlocConv(64, 64),
                 BlocConv(64, 128), ConvDown(128, 128), BlocConv(128, 128), nn.Flatten(), nn.Linear(128 * 49, 128), nn.Linear(128, 10))

    def forward(self, x):
              
        return self.net(x)
    

from tqdm import tqdm
import os
os.makedirs("sample", exist_ok=True)

def train(n_époque=100):

    modèle = Modèle()
    critère = nn.CrossEntropyLoss()
    optimiseur = torch.optim.AdamW(modèle.parameters(), lr=5e-4)

    def train_step(batch):
        images, tags = batch

        logits = modèle(images)

        perte = critère(logits, tags)

        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()

        prédictions = torch.argmax(logits, dim=1)
        bonnes_prédictions = (prédictions == tags).sum().item()

        return perte.item(), bonnes_prédictions, tags.size(0)

    barre_époque = tqdm(range(n_époque), desc='Entraînement', leave=True)
    
    for époque in barre_époque:
        batches = tqdm(loader, desc=f"Époque {époque + 1}", leave=False)

        perte_total = 0
        bonnes_total = 0
        total = 0

        for batch in batches:
            perte_batch, bonnes_prédictions, n = train_step(batch)

            perte_total += perte_batch
            bonnes_total += bonnes_prédictions
            total += n

            batches.set_postfix({
                "Perte lot": f"{perte_batch:.6f}",
                "Acc lot": f"{bonnes_prédictions/n:.4f}"
            })

        perte_train = perte_total/len(loader)
        précision_train = bonnes_total/total
        
        barre_époque.set_postfix({
            "Perte époque": f"{perte_train:.6f}",
            "Acc époque": f"{précision_train:.4f}"
        })
        
        with torch.no_grad():
            test = tqdm(loader_test, desc=f"Val {époque + 1}", leave= False)

            perte_total = 0
            bonnes_total = 0
            total = 0

            for batch in test:
                images, tags = batch

                logits = modèle(images)
                perte = critère(logits, tags)

                perte_total += perte.item()

                prédictions = torch.argmax(logits, dim=1)
                bonnes = (prédictions == tags).sum().item()

                bonnes_total += bonnes
                total += tags.size(0)

                test.set_postfix({
                    "perte val": f"{perte.item():.6f}",
                    "acc val": f"{bonnes / tags.size(0):.4f}"
                })

            perte_val = perte_total / len(loader_test)
            acc_val = bonnes_total / total
            
            barre_époque.set_postfix({
                "Perte val": f"{perte_val:.6f}",
                "Acc val": f"{acc_val:.4f}"
            })

            tqdm.write(
            f"Époque {époque + 1:>3}/{n_époque}"
            f"  │  perte train: {perte_train:.4f}  acc train: {précision_train:.4f}"
            f"  │  perte val:   {perte_val:.4f}  acc val:   {acc_val:.4f}"
        )   
            fig, axes = plt.subplots(2, 5, figsize = (12,8))

            batch, tags = next(iter(loader_test))

            for i in range(10):
                img = batch[i]

                logits = modèle(img.unsqueeze(0)).squeeze(0)
                prédictions = torch.argmax(logits).item()

                img_np = img.permute(1, 2, 0).detach()

                axes[i//5, i%5].set_title(f"prédictions : {str(prédictions)}   Vrai: {tags[i].squeeze().item()}")
                axes[i//5, i%5].imshow(img_np, cmap='gray')
                axes[i//5, i%5].axis('off')

            plt.savefig(f"./sample/sample_{époque + 1}")
            plt.close()
                
    torch.save(modèle.state_dict(), "checkpoint_FashionMNIST.pth")
    

train()      
        
        
        
