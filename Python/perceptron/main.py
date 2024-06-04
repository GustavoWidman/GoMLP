from datetime import datetime

import torch
from utils.math import round_tensor
from utils.text import format_tensors_side_by_side


class Perceptron:
	def __init__(self, inputRows: int, inputCols: int, learningRate: float):
		self.model = torch.nn.Sequential(
			torch.nn.Linear(inputCols, inputRows),
			torch.nn.ReLU(),
			torch.nn.Linear(inputRows, 1),
			torch.nn.Sigmoid()
		)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningRate)
		self.loss_fn = torch.nn.BCELoss()

	def feed_forward(self, input: torch.Tensor):
		return self.model(input) # WOW THATS EASY

	def back_propagation(self, input: torch.Tensor, target: torch.Tensor):
		prediction = self.feed_forward(input)

		# Calculate how wrong the prediction was
		err = self.loss_fn(prediction, target)
		err.backward()

		# Update weights
		self.optimizer.step()
		self.optimizer.zero_grad()

		return err.item()

	def train(self, input: torch.Tensor, output: torch.Tensor, epochs: int):
		print(f'\nTraining for {epochs} epochs...')

		start = datetime.now()
		for epoch in range(epochs):
			loss = self.back_propagation(input, output)

			if epoch < 2000:
				if epoch%500 == 0:
					predictions = self.feed_forward(input)

					accuracy = self.accuracy(predictions, output)

					print(f'Epoch: {epoch} 	Loss: {loss:.4f} 	Accuracy: {accuracy:.2f}%')

			elif epoch%5000 == 0:
				predictions = self.feed_forward(input)

				accuracy = self.accuracy(predictions, output)

				print(f'Epoch: {epoch} 	Loss: {loss:.4f} 	Accuracy: {accuracy:.2f}%')

		print(f'Training took {round((datetime.now() - start).total_seconds(), 2)}s')

	def accuracy(self, predicted: torch.Tensor, expected: torch.Tensor):
		accuracy = []

		for i in range(len(predicted)):
			for j in range(len(predicted[i])):
				diff = abs(float(predicted[i][j]) - float(expected[i][j]))
				accuracy.append(1 - diff)

		return sum(accuracy) / len(accuracy) * 100

	def benchmark(self, input: torch.Tensor, output: torch.Tensor):
		predictions = self.feed_forward(input)
		rounded_predictions = round_tensor(predictions)

		accuracy = self.accuracy(predictions, output)
		rounded_accuracy = self.accuracy(rounded_predictions, output)

		print("Predictions (raw vs rounded):")
		print(format_tensors_side_by_side(predictions, rounded_predictions))

		print(f"Accuracy: {round(accuracy, 2)}%")
		print(f"Rounded accuracy: {round(rounded_accuracy, 2)}%")