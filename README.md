# 🧠 Synthetic Market Simulator using GANs + Backtrader

This project leverages Generative Adversarial Networks (GANs) to create synthetic financial time-series data for robust strategy development and backtesting using Backtrader.

## 🚀 Features

- Train a GAN to model historical stock market patterns.
- Generate synthetic market data with statistical realism.
- Backtest trading strategies (e.g., RSI) on synthetic datasets.
- Evaluate robustness across alternate "market realities".

## 🛠️ Tech Stack

- **PyTorch** – for building and training the GAN.
- **yfinance** – to download historical market data.
- **scikit-learn** – for data preprocessing and scaling.
- **Backtrader** – for strategy simulation and backtesting.
- **Matplotlib / Pandas** – for data visualization and analysis.

## 🧪 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main pipeline:

```bash
python main.py
```

3. View results:

- Synthetic data: `synthetic_data.csv`
- Training loss plot: `results/training_loss.png`
- Backtest results: Displayed via Backtrader plot.

## 📊 Example Use Case

```python
from strategy import RSI_Strategy

# Plug in different strategies into Backtrader to test on synthetic data
```

## 📄 License

MIT License

---

## Author

Rohan G
