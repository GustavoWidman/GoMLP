package utils

import (
	"fmt"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Helper function to format matrices side by side
func FormatMatricesSideBySide(m1, m2 *mat.Dense) string {
	r1, c1 := m1.Dims()
	r2, c2 := m2.Dims()

	if r1 != r2 {
		return "Matrices have different row counts and cannot be formatted side by side"
	}

	var result strings.Builder

	// Format the matrices side by side
	for i := 0; i < r1; i++ {
		for j := 0; j < c1; j++ {
			result.WriteString(fmt.Sprintf("   %4f   ", m1.At(i, j)))
		}
		result.WriteString("|")
		for j := 0; j < c2; j++ {
			result.WriteString(fmt.Sprintf("   %4f   ", m2.At(i, j)))
		}
		result.WriteString("\n")
	}

	return result.String()
}
