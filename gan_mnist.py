import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

x = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
)

x_data_loader = torch.utils.data.DataLoader(
    x,
    batch_size=128,
    shuffle=True
)

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, 784)
        return self.nn(x_flat)

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        x_g_flat = self.nn(z)
        # Usar -1 no .view() do PyTorch faz com que ele calcule o tamanho do
        # batch automaticamente com base no que sobrou dos dados.
        # Evita problemas caso o último lote seja menor do que o normal.
        return x_g_flat.view(-1, 1, 28, 28)

d = D().to(device)
g = G().to(device)

loss_function = nn.BCELoss()

optimizer_d = optim.Adam(d.parameters(), lr=0.0002)
optimizer_g = optim.Adam(g.parameters(), lr=0.0002)

epochs = 50
fixed_noise = torch.randn((9, 100)).to(device)

epoch_iterator = tqdm(range(epochs), desc="Training GAN", unit="epoch")

for epoch in epoch_iterator:
    # Discard original labels from MNIST with _
    for x_batch, _ in x_data_loader:
        x_batch_size = x_batch.size(0)

        # Move the batch to GPU
        x_batch = x_batch.to(device)

        x_labels = torch.ones((x_batch_size, 1)).to(device)
        z_labels = torch.zeros((x_batch_size, 1)).to(device)
        z = torch.randn((x_batch_size, 100)).to(device)

        # Training the discriminator
        d.train()
        optimizer_d.zero_grad()

        # Passing real images to D
        x_predictions = d(x_batch)
        x_loss = loss_function(x_predictions, x_labels)

        # Passing fake images to D (isolating the G gradient)
        z_predictions = d(g(z).detach())
        z_loss = loss_function(z_predictions, z_labels)

        # Updating D weights
        total_loss = x_loss + z_loss
        total_loss.backward()
        optimizer_d.step()

        # Training the generator
        g.train()
        optimizer_g.zero_grad()

        # Passing fake images (without discarding the G gradient)
        # Evaluating the error against the real image labels 
        z_predictions = d(g(z))
        g_loss = loss_function(z_predictions, x_labels)

        # Updating the weights
        g_loss.backward()
        optimizer_g.step()

    epoch_iterator.set_postfix({"D_Loss": f"{total_loss.item():.4f}", "G_Loss": f"{g_loss.item():.4f}"})

    if (epoch + 1) % 10 == 0:
        g.eval()
        with torch.no_grad():
            g_images = g(fixed_noise)
            g_images =(g_images + 1.0) / 2.0
            file_path = f"./images/gan_epoch_progress_{epoch+1}.png"
            utils.save_image(g_images, file_path, nrow=3)
            tqdm.write(f"Image progress grid saved in: {file_path}")
