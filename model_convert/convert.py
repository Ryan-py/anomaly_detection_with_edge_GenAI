import torch
import torch.nn as nn
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.utils.configs import get_transform_passes
from executorch.exir import to_edge_transform_and_lower

# 1. Deterministic Inference Architecture
class VAE_Edge_Inference(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.flatten_size = 256 * 14 * 14
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        z = self.decoder_input(mu)
        z = z.view(-1, 256, 14, 14)
        reconstruction = self.decoder(z)
        return reconstruction

print("1. Loading trained model weights...")
model = VAE_Edge_Inference(latent_dim=512).to("cpu")
model.load_state_dict(torch.load("vae_edge_weights.pth", weights_only=True))
model.eval()

print("2. Loading calibration dataset...")
calib_tensor = torch.load("calibration_data.pt", weights_only=True)
sample_inputs = (calib_tensor[0:1],)  # Shape: (1, 3, 224, 224)

print("3. Configuring XNNPACK Quantizer...")
qparams = get_symmetric_quantization_config(is_per_channel=True)
quantizer = XNNPACKQuantizer()
quantizer.set_global(qparams)

print("4. Preparing model for PT2E quantization...")
training_ep = torch.export.export(model, sample_inputs).module()
prepared_model = prepare_pt2e(training_ep, quantizer)

print("5. Calibrating activation ranges...")
with torch.no_grad():
    for i in range(calib_tensor.shape[0]):
        prepared_model(calib_tensor[i:i+1])

print("6. Converting to INT8 quantized model...")
quantized_model = convert_pt2e(prepared_model)

print("7. Lowering with XnnpackPartitioner and transform passes...")
et_program = to_edge_transform_and_lower(
    torch.export.export(quantized_model, sample_inputs),
    partitioner=[XnnpackPartitioner()],
    transform_passes=get_transform_passes(),
).to_executorch()

print("8. Saving binary to disk...")
output_filename = "vae_anomaly_xnnpack_int8.pte"
with open(output_filename, "wb") as f:
    f.write(et_program.buffer)

print(f"Compilation finished successfully: {output_filename}")