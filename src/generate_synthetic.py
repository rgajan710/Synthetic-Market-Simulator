import torch
import numpy as np
import pandas as pd
from gan_model import Generator

def generate_synthetic_samples(generator, num_samples, window_size=60, latent_dim=100, device='cpu'):
    generator.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, window_size, latent_dim).to(device)
            gen_seq = generator(z).squeeze(0).cpu().numpy()
            samples.append(gen_seq)

    return np.array(samples)

def save_synthetic_to_csv(samples, scaler, filename="synthetic_data.csv", start_date="2020-01-01", freq='1D'):
    all_samples = []
    current_date = pd.to_datetime(start_date)

    for sample in samples:
        rescaled = scaler.inverse_transform(sample)
        df = pd.DataFrame(rescaled, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Generate datetime index
        dates = pd.date_range(current_date, periods=len(df), freq=freq)
        df['timestamp'] = dates
        current_date = dates[-1] + pd.Timedelta(days=1)  # Avoid overlap

        all_samples.append(df)

    full_df = pd.concat(all_samples, axis=0)
    full_df.set_index('timestamp', inplace=True)
    full_df.to_csv(filename)