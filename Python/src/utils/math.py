import torch

def round_tensor(tensor: torch.Tensor):
	r, c = tensor.shape
	rounded = torch.zeros(r, c)

	for i in range(r):
		for j in range(c):
			rounded[i][j] = round(tensor[i][j].item())

	return rounded