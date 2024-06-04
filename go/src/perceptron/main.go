package perceptron

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"

	mathUtil "go-perceptron/src/math"
	"go-perceptron/src/utils"
)

// Perceptron struct that contains the weights and the activation thresholds
type Perceptron struct {
	wi     *mat.Dense
	wh     *mat.Dense
	hidden *mat.Dense
	output *mat.Dense
}

// NewPerceptron initializes a new Perceptron given the number of rows and columns of the input matrix
func NewPerceptron(inputRows, inputCols int) *Perceptron {
	perceptron := Perceptron{
		wi: mat.NewDense(inputCols, inputRows, nil), // weights for input to hidden layer
		wh: mat.NewDense(inputRows, 1, nil),         // weights for hidden to output layer
	}

	perceptron.InitializeWeights()

	return &perceptron
}

// InitializeWeights initializes weights with random values
func (p *Perceptron) InitializeWeights() {
	p.wi = mathUtil.RandomWeights(p.wi)
	p.wh = mathUtil.RandomWeights(p.wh)
}

// FeedForward performs feedforward on the neural network, returning the output layer
func (p *Perceptron) FeedForward(input *mat.Dense) *mat.Dense {
	inputRows := input.RawMatrix().Rows

	// Hidden layer activation
	hidden := mat.NewDense(inputRows, p.wi.RawMatrix().Cols, nil)
	hidden.Mul(input, p.wi)
	hidden = mathUtil.Apply(hidden, mathUtil.Sigmoid)
	p.hidden = hidden

	// Output layer activation
	output := mat.NewDense(inputRows, p.wh.RawMatrix().Cols, nil)
	output.Mul(p.hidden, p.wh)
	output = mathUtil.Apply(output, mathUtil.Sigmoid)
	p.output = output

	return output
}

// BackPropagation performs backpropagation on the neural network, updating the weights based on the error
func (p *Perceptron) BackPropagation(input, target *mat.Dense, learningRate float64) {
	predictions := p.FeedForward(input)
	r, c := target.Dims()

	// Calculate how wrong the prediction was
	err := mat.NewDense(r, c, nil)
	err.Sub(target, predictions)

	// Calculate the delta for the output layer
	deltaOutput := mathUtil.Apply(predictions, mathUtil.SigmoidDerivative)
	deltaOutput.MulElem(deltaOutput, err)
	deltaOutputCols := deltaOutput.RawMatrix().Cols

	// Calculate the delta for the hidden layer
	hiddenRows, hiddenCols := p.hidden.Dims()
	errorHidden := mat.NewDense(hiddenRows, hiddenCols, nil)
	errorHidden.Mul(deltaOutput, p.wh.T())

	deltaHidden := mathUtil.Apply(p.hidden, mathUtil.SigmoidDerivative)
	deltaHidden.MulElem(deltaHidden, errorHidden)

	// Update weights: hidden to output layer
	hiddenTrans := mat.DenseCopyOf(p.hidden.T())
	deltaOutputScaled := mat.NewDense(hiddenCols, deltaOutputCols, nil)
	deltaOutputScaled.Mul(hiddenTrans, deltaOutput)
	deltaOutputScaled.Scale(learningRate, deltaOutputScaled)
	p.wh.Add(p.wh, deltaOutputScaled)

	// Update weights: input to hidden layer
	inputsTrans := mat.DenseCopyOf(input.T())
	deltaHiddenScaled := mat.NewDense(inputsTrans.RawMatrix().Rows, deltaHidden.RawMatrix().Cols, nil)
	deltaHiddenScaled.Mul(inputsTrans, deltaHidden)
	deltaHiddenScaled.Scale(learningRate, deltaHiddenScaled)
	p.wi.Add(p.wi, deltaHiddenScaled)
}

// Train trains the neural network by iterating over the dataset for a number of epochs
func (p *Perceptron) Train(inputs, outputs *mat.Dense, learningRate float64, epochs int) {
	defer utils.Time(utils.Config{Name: "Training"})()

	fmt.Printf("\nTraining for %d epochs...\n", epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		p.BackPropagation(inputs, outputs, learningRate)

		// This is just for demonstration purposes
		// May slightly slow down the training process
		if epoch < 2000 {
			if epoch%500 == 0 {
				predictions := p.FeedForward(inputs)

				accuracy := p.Accuracy(predictions, outputs)

				fmt.Printf("Epoch %d 	Accuracy: %.2f%%\n", epoch, accuracy)
			}
		} else if epoch%50000 == 0 {
			predictions := p.FeedForward(inputs)

			accuracy := p.Accuracy(predictions, outputs)

			fmt.Printf("Epoch %d 	Accuracy: %.2f%%\n", epoch, accuracy)
		}
	}
}

// Accuracy calculates the accuracy of the network
func (p *Perceptron) Accuracy(predicted, expected *mat.Dense) float64 {
	r, c := predicted.Dims()
	accuracy := []float64{}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			diff := math.Abs(predicted.At(i, j) - expected.At(i, j))
			accuracy = append(accuracy, 1-diff)
		}
	}
	return mathUtil.Average(accuracy) * 100
}

func (p *Perceptron) Benchmark(inputs, outputs *mat.Dense) {
	predictions := p.FeedForward(inputs)
	roundedPredictions := mathUtil.RoundMatrix(predictions)

	accuracy := p.Accuracy(predictions, outputs)
	roundedAccuracy := p.Accuracy(roundedPredictions, outputs)

	fmt.Println("Predictions (raw vs rounded):")
	fmt.Println(utils.FormatMatricesSideBySide(predictions, roundedPredictions))

	fmt.Printf("Accuracy: %.2f%%\n", accuracy)
	fmt.Printf("Rounded accuracy: %.2f%%\n", roundedAccuracy)
}
