package math

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Initialize a matrix with random Float64 values all across the matrix
func RandomWeights(matrix *mat.Dense) *mat.Dense {
	r, c := matrix.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			matrix.Set(i, j, rand.Float64())
		}
	}
	return matrix
}

// Apply a function to each element of a matrix
func Apply(matrix *mat.Dense, f func(float64) float64) *mat.Dense {
	r, c := matrix.Dims()
	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, f(matrix.At(i, j)))
		}
	}
	return result
}

// Scale matrix by a factor k
func Scale(k float64, matrix *mat.Dense) *mat.Dense {
	r, c := matrix.Dims()
	scaled := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			scaled.Set(i, j, matrix.At(i, j)*k)
		}
	}
	return scaled
}

// Sigmoid function
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid derivative function
func SigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}

// A utility function to round the predictions to the nearest integer (0 or 1).
func RoundMatrix(matrix *mat.Dense) *mat.Dense {
	r, c := matrix.Dims()
	rounded := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			rounded.Set(i, j, math.Round(matrix.At(i, j)))
		}
	}
	return rounded
}

// Calculate the average of a list of float64 values
func Average(list []float64) float64 {
	sum := 0.0
	for _, val := range list {
		sum += val
	}
	return sum / float64(len(list))
}
