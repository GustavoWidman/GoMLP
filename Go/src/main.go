package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"go-perceptron/src/perceptron"
)

func main() {
	inputs := mat.NewDense(4, 2, []float64{
		0, 0, // XOR f-f -> f
		0, 1, // XOR f-t -> t
		1, 0, // XOR t-f -> t
		1, 1, // XOR t-t -> f
	})

	outputs := mat.NewDense(4, 1, []float64{
		0,
		1,
		1,
		0,
	})

	rows, cols := inputs.Dims()
	p := perceptron.NewPerceptron(rows, cols)

	fmt.Println("Benchmark before training:")
	p.Benchmark(inputs, outputs)

	p.Train(inputs, outputs, 0.05, 50000)

	fmt.Println("\nBenchmark after training:")
	p.Benchmark(inputs, outputs)
}
