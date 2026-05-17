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
        

class Learping(nn.Module):
    def __init__(self, m1, m2, n):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.alpha = nn.Parameter(torch.zeros(1, n))

    def forward(self, x):
        w = self.alpha.sigmoid()
        return self.m1(x) * (1 - w) + self.m2(x) * w
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1_1 = nn.Sequential(nn.Linear(784, 128), nn.SiLU())
        self.c1_2 = nn.Sequential(nn.Linear(784, 128), nn.SiLU())
        self.lerp1 = Learping(self.c1_1, self.c1_2, 128)

        self.c2_1 = nn.Sequential(nn.Linear(128, 64), nn.SiLU())
        self.c2_2 = nn.Sequential(nn.Linear(128, 64), nn.SiLU())
        self.lerp2 = Learping(self.c2_1, self.c2_2, 64)

        self.c3_1 = nn.Sequential(nn.Linear(64, 10), nn.SiLU())
        self.c3_2 = nn.Sequential(nn.Linear(64, 10), nn.SiLU())
        self.lerp3 = Learping(self.c3_1, self.c3_2, 10)


    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.lerp1(x)
        x = self.lerp2(x)
        x = self.lerp3(x)
        return x
    

from tqdm import tqdm


import os
os.makedirs("sample", exist_ok=True)

def train(n_epoch=100):

    modèle = Model()
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

    bar_epoch = tqdm(range(n_epoch), desc='Entraînement', leave=False)

    for epoch in bar_epoch:
        batches = tqdm(loader, desc=f"Époque {epoch + 1}")

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
                "Acc": f"{bonnes_pred/n:.4f}"
            })

        bar_epoch.set_postfix({
            "Perte époque": f"{perte_total/len(loader):.6f}",
            "Acc époque": f"{bonnes_total/total:.4f}"
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

            bar_epoch.set_postfix({
                "Perte val": f"{perte_total / len(loader_test):.6f}",
                "Acc val": f"{bonnes_total / total:.4f}"
            })

                
                
            fig, axes = plt.subplots(1, 5)

            batch, _ = next(iter(loader_test))

            for i in range(5):
                img = batch[i]

                logits = modèle(img.unsqueeze(0)).squeeze(0)
                pred = torch.argmax(logits).item()

                img_np = img.permute(1, 2, 0).detach().cpu()

                axes[i].set_title(str(pred))
                axes[i].imshow(img_np, cmap='gray')

            plt.savefig(f"./sample/sample_{epoch + 1}")
                
            
    

train()      
        
        
        
