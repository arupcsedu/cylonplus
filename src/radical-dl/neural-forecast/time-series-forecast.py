from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.utils import AirPassengersDF
import torch

print("Running")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
    
nf = NeuralForecast(
    models = [NBEATS(input_size=24, h=12, max_steps=100, accelerator="gpu")],
    freq = 'M',
)
print("NF setup", nf)
nf.fit(df=AirPassengersDF)
nf.predict()