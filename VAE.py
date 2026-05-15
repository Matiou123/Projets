import torch
import torch.nn as nn
import torch.nn.functional as F


class BlocRésiduelle(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()

        self.connection = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size = 1)
        self.bloc1 = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.GroupNorm(32, c_out), nn.SiLU())
        self.bloc2 = nn.Sequential(nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), nn.GroupNorm(32, c_out), nn.SiLU())

    def forward(self, x):
        x_copie = x
        x = self.bloc1(x)
        x= self.bloc2(x)

        return F.silu(x + self.connection(x_copie))
    
class Encodeur(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.SiLU())

        self.res1 = BlocRésiduelle(64,64)

        self.res2 = BlocRésiduelle(64, 128)

        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride= 2, padding=1), nn.GroupNorm(32, 128), nn.SiLU())
        
        self.res3 = BlocRésiduelle(128,128)

        self.res4 = BlocRésiduelle(128, 256)

        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride= 2, padding=1), nn.GroupNorm(32, 256), nn.SiLU())

        
        self.res5 = BlocRésiduelle(256,256)

        self.res6 = BlocRésiduelle(256, 512)

        self.conv4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride= 2, padding=1), nn.GroupNorm(32, 512), nn.SiLU())

        self.res7 = BlocRésiduelle(512,512)

        self.conv5= nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.conv3(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.conv4(x)
        x = self.res7(x)
        x = self.conv5(x)

        mu, log_var = torch.chunk(x, 2, dim = 1)

        log_var = torch.clamp(log_var, -30, 20)

        std =  torch.exp(0.5 * log_var)

        z = mu + std * torch.randn_like(std)

        return z, mu, log_var


class Decodeur(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.GroupNorm(32, 512), nn.SiLU())
        
        self.res1 = BlocRésiduelle(512,512)
        self.res2 = BlocRésiduelle(512,256)

        
        self.tconv1 = nn.Sequential(nn.Upsample(scale_factor=2), BlocRésiduelle(256,256),nn.GroupNorm(32, 256), nn.SiLU())

        self.res3 = BlocRésiduelle(256,256)
        self.res4 = BlocRésiduelle(256,128)

        self.tconv2 = nn.Sequential(nn.Upsample(scale_factor=2), BlocRésiduelle(128,128), nn.GroupNorm(32, 128), nn.SiLU())

        self.res5 = BlocRésiduelle(128,128)
        self.res6 = BlocRésiduelle(128,64)

        self.tconv3 = nn.Sequential(nn.Upsample(scale_factor=2), BlocRésiduelle(64,64), nn.GroupNorm(32, 64), nn.SiLU())

        self.res7 = BlocRésiduelle(64,64)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.tconv1(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.tconv2(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.tconv3(x)
        x = self.res7(x)
        x = self.conv2(x)

        return torch.sigmoid(x)
    
class VAE(nn.Module):

    def __init__(self, ):
        super().__init__()

        self.enc = Encodeur()
        self.dec = Decodeur()

    def forward(self, x):
        z , mu, log_var  = self.enc(x)
        return self.dec(z), mu, log_var 
    
    @torch.no_grad()
    def encode(self,x):
        _, mu, _ = self.enc(x)
        return mu
    
    @torch.no_grad()
    def decode(self,x):
        return self.dec(x)
