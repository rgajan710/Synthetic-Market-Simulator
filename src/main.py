import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt

from data_loader import download_data, preprocess_data
from gan_model import Generator, Discriminator
from generate_synthetic import generate_synthetic_samples, save_synthetic_to_csv
from strategy import RSI_Strategy

def train_gan(real_data, latent_dim=100, num_epochs=300, device='cpu'):
    seq_len, feature_dim = real_data.shape[1], real_data.shape[2]
    
    G = Generator(latent_dim, 128, feature_dim).to(device)
    D = Discriminator(feature_dim, 128).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

    losses_G, losses_D = [], []

    for epoch in range(num_epochs):
        idx = np.random.randint(0, real_data.shape[0], 64)
        real_seq = torch.tensor(real_data[idx], dtype=torch.float32).to(device)

        valid = torch.ones((64, 1)).to(device)
        fake = torch.zeros((64, 1)).to(device)

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(64, seq_len, latent_dim).to(device)
        generated_seq = G(z)
        g_loss = criterion(D(generated_seq), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(D(real_seq), valid)
        fake_loss = criterion(D(generated_seq.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        losses_G.append(g_loss.item())
        losses_D.append(d_loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    os.makedirs("results", exist_ok=True)
    plt.plot(losses_D, label='Discriminator Loss')
    plt.plot(losses_G, label='Generator Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.savefig("results/training_loss.png")
    plt.close()

    os.makedirs("models", exist_ok=True)
    torch.save(G.state_dict(), "models/generator.pth")
    torch.save(D.state_dict(), "models/discriminator.pth")
    print("[INFO] Models saved to models/ folder.")

    return G
    
def run_backtest(csv_file):
    # Read the CSV with timestamp parsing
    df = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)

    # Feed data to Backtrader
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(RSI_Strategy)
    cerebro.adddata(data)
    cerebro.broker.set_cash(100000)
    cerebro.run()
    cerebro.plot()
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Load and preprocess real data
    df = download_data()
    real_windows, scaler = preprocess_data(df)

    # Step 2: Train GAN
    generator = train_gan(real_windows, device=device)

    # Step 3: Generate synthetic data
    synthetic_data = generate_synthetic_samples(generator, num_samples=10, device=device)
    save_synthetic_to_csv(synthetic_data, scaler)

    # Step 4: Run backtest
    run_backtest("synthetic_data.csv")
