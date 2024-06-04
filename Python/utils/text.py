import torch


def format_tensors_side_by_side(t1: torch.Tensor, t2: torch.Tensor) -> str:
	r1, c1 = t1.shape
	r2, c2 = t2.shape

	if r1 != r2:
		return "Tensors have different row counts and cannot be formatted side by side"

	result = ""

	# Format the tensors side by side
	for i in range(r1):
		for j in range(c1):
			result += f"   {format(float(t1[i][j]), '.6f')}   "
		result += "|"
		for j in range(c2):
			result += f"   {format(float(t2[i][j]), '.6f')}   "
		result += "\n"

	return result