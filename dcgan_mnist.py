import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

#hiperparametros

batch_size = 512  
channels = 1      
image_size = 64   
z_dim = 100
features_d = 64
features_g = 64
learning_rate = 0.0002
beta1 = 0.5
epochs = 25      

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando na GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

#dataset

dataset = dset.MNIST(root="./data", download=True,
                     transform=transforms.Compose([
                         transforms.Resize(image_size),
                         transforms.ToTensor(),
                         # para 1 canal
                         transforms.Normalize((0.5,), (0.5,)),
                     ]))


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"--> Dataset carregado com {len(dataset)} imagens prontas para treino!")


#arquitetura de um canal
 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

netG = Generator().to(device)
netD = Discriminator().to(device)

#configuracao de loss e otimizadores 
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))



#treinamento e preparamento
fixed_noise = torch.randn(9, z_dim, 1, 1, device=device) # ver os mesmos numeros durante  evolucao

# pra nao misturar as coisas na pasta
os.makedirs("resultados_mnist", exist_ok=True)

print("\nIniciando o Treinamento da DCGAN no MNIST...")

for epoch in range(epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
    for i, data in loop:
        
        real_img = data[0].to(device)
        b_size = real_img.size(0)
        
        real_label = torch.ones((b_size,), dtype=torch.float, device=device)
        fake_label = torch.zeros((b_size,), dtype=torch.float, device=device)

        # treina discriminador
        netD.zero_grad()
        output_real = netD(real_img).view(-1)
        loss_D_real = criterion(output_real, real_label)
        loss_D_real.backward()
        
        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        fake_img = netG(noise)
        
        output_fake = netD(fake_img.detach()).view(-1)
        loss_D_fake = criterion(output_fake, fake_label)
        loss_D_fake.backward()
        optimizerD.step()

        # treina o gerador
        netG.zero_grad()
        output = netD(fake_img).view(-1)
        loss_G = criterion(output, real_label)
        loss_G.backward()
        optimizerG.step()
        
        # isso e pra atualizar as epocas no terminal
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss_D=(loss_D_real + loss_D_fake).item(), loss_G=loss_G.item())

    #salvar tudo

    netG.eval()
    with torch.no_grad():
        fake_images = netG(fixed_noise).detach().cpu()
    netG.train()

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(f"Dígitos Falsos (MNIST) - Época {epoch+1}")
    
    # normalize= true pros cinzas ficarem bons
    grid_img = vutils.make_grid(fake_images, nrow=3, padding=2, normalize=True).permute(1,2,0)
    plt.imshow(grid_img, cmap='gray')

    # salva dentro da pasta
    caminho_arquivo = os.path.join("resultados_mnist", f"mnist_epoca_{epoch+1}.png")
    plt.savefig(caminho_arquivo, bbox_inches='tight')
    plt.close() # <---- pra memoria ram
print("\n--> Treinamento concluído com sucesso!")
print("--> Verifique a pasta 'resultados_mnist' na barra lateral do VS Code para ver a evolução.")

# salva peso treinado pra gerar futuramente
torch.save(netG.state_dict(), "gerador_mnist.pth")
print("--> O modelo treinado foi salvo como 'gerador_mnist.pth'.")