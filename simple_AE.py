import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
mnist = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=True, download=False, transform=transform)


loader = DataLoader(mnist, batch_size= 64, shuffle = True)
loader_test = DataLoader(mnist_test, batch_size= 64, shuffle = True)


class Encodeur(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.enc = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.SiLU(),
                                 nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU())
    def forward(self, x):
        return self.enc(x)
        
class Décodeur(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.déc = nn.Sequential(nn.ConvTranspose2d(64, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(16), nn.SiLU(),
                                 nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.SiLU())
    def forward(self, x):
        return self.déc(x)        

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.enc = Encodeur()
        
        self.dec = Décodeur()
        
    def forward(self, x):
        return self.dec(self.enc(x))
    
    def encode(self, x): return self.enc(x)
    
    def decode(self, x): return self.dec(x)
    

from tqdm import tqdm


import os
os.makedirs("sample", exist_ok=True)

def train(n_epoch = 10):

    
    modèle = Model()
    critère = nn.MSELoss()
    optimiseur = torch.optim.AdamW(modèle.parameters(), lr = 5e-4)
    
    
    
    def train_step(batch):
        images , _ = batch
        
        recons = modèle(images)
        
        perte = critère(recons, images)
        
        optimiseur.zero_grad()
        perte.backward()
        optimiseur.step()
        
        
        return perte/images.shape[0]
    
    bar_epoch= tqdm(range(n_epoch), desc='Entraînement')
    
    for epoch in bar_epoch:
        batches = tqdm(loader, desc= f"Époque {epoch + 1}")
        perte_total = 0
        
        for batch in batches:
            perte_batch = 0
            
            perte_batch = train_step(batch)
            perte_total += perte_batch
        
            batches.set_postfix({"Perte du lot":  f"{perte_batch:.7f}"})
        
        bar_epoch.set_postfix({"Perte époque" : f"{perte_total/len(batches):.7f}"})
        
        with torch.no_grad():
            test = tqdm(loader_test, desc=f"Val {epoch + 1}")
            perte_total = 0
            for batches in loader_test:
                perte_batch = 0
                images, _ = batches
                recons = modèle(images)
                
                perte = critère(recons, images)
                
                perte_batch = perte/images.shape[0]
                test.set_postfix({"perte val" : f"{perte_batch:.7f}"})
                perte_total += perte_batch
            
            bar_epoch.set_postfix({"Perte val" : f"{perte_total/len(batches):.7f}"})

            
            
            fig, axes = plt.subplots(2,5)
            bacth, _ = next(iter(loader_test))
            for i in range(5):
                img = bacth[i]
                recons = modèle(img.unsqueeze(0)).squeeze(0)
                img = img.permute(1,2,0).detach()
                recons = recons.permute(1,2,0).detach()
                axes[0][i].imshow(img, cmap='gray')
                axes[1][i].imshow(recons, cmap='gray')
                

            plt.savefig(f"./sample/sample_{epoch + 1}")
                
            
    

train() 
