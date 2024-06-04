import torch
from perceptron.main import Perceptron


def main():
	inputs = torch.tensor([
		[0, 0], # XOR f-f -> f
		[0, 1], # XOR f-t -> t
		[1, 0], # XOR t-f -> t
		[1, 1]  # XOR t-t -> f
	], dtype=torch.float32)

	outputs = torch.tensor([
		[0],
		[1],
		[1],
		[0]
	], dtype=torch.float32)

	p = Perceptron(inputs.shape[0], inputs.shape[1], 0.03)

	print("Benchmark before training:")
	p.benchmark(inputs, outputs)

	p.train(inputs, outputs, 50000)

	print("\nBenchmark after training:")
	p.benchmark(inputs, outputs)

if __name__ == "__main__":
	main()