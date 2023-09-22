import torch
import torch.nn as nn
import torch.optim as optim

from manageFiles import *

# Definizione del modello VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # mu = la media della distribuzione latente
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # logvar = logaritmo della varianza

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        if x.shape > 1:
            x = torch.flatten(x)
        h1 = torch.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



# Funzione di perdita (Loss)
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

'''# Parametri del modello e ottimizzatore
input_dim = 1280  # Numero di note MIDI possibili (0-127)
hidden_dim = 64  # Dimensione nascosta
latent_dim = 16  # Dimensione latente

model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Allenamento del modello (esempio: dati MIDI appiattiti)
num_epochs = 100



model = loadVariableFromFile("./variables/model")
# Generazione di campioni MIDI
with torch.no_grad():
    sample = torch.randn(1, latent_dim)  # Genera un campione dalla dimensione latente
    sample = model.decode(sample).cpu().numpy()  # Decodifica e converti in formato MIDI


    # Ora 'sample' contiene un campione MIDI generato dal VAE'''