from torchvision.models import vit_b_16
import torch

# Initialize the model
model = torch.hub.load("pytorch/vision", "vit_b_16", weights="IMAGENET1K_V1")


# keys = list(model.state_dict().keys())
# print("Keys:", keys)
# Define arguments
# args = {
#     'model_args': {'gate_scale': 10.0, 'gate_center': 75.0},
#     'training_args': {'learning_rate': 0.001, 'batch_size': 512, 'epochs': 100}
# }

# Save the model and arguments
save_path = "/home/jvincenti/D2DMoE/shared/results/effbench_runs/TINYIMAGENET_PATH_mha_rep_distill_J4CULVA3_1/final.pth"
torch.save(model.state_dict(), save_path)
