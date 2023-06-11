// Copyright 2023 The Meta Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/datum/iris"

	"gonum.org/v1/exp/linsolve"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// System represents a linear system with a symmetric band matrix
//
//	A*x = b
type System struct {
	A *mat.SymBandDense
	B *mat.VecDense
}

// L2Projection returns a linear system whose solution is the L2 projection of f
// into the space of piecewise linear functions defined on the given grid.
//
// References:
//   - M. Larson, F. Bengzon, The Finite Element Method: Theory,
//     Implementations, and Applications. Springer (2013), Section 1.3, also
//     available at:
//     http://www.springer.com/cda/content/document/cda_downloaddocument/9783642332869-c1.pdf
func L2Projection(grid []float64, f func(float64) float64) System {
	n := len(grid)

	// Assemble the symmetric banded mass matrix by iterating over all elements.
	A := mat.NewSymBandDense(n, 1, nil)
	for i := 0; i < n-1; i++ {
		// h is the length of the i-th element.
		h := grid[i+1] - grid[i]
		// Add contribution from the i-th element.
		A.SetSymBand(i, i, A.At(i, i)+h/3)
		A.SetSymBand(i, i+1, h/6)
		A.SetSymBand(i+1, i+1, A.At(i+1, i+1)+h/3)
	}

	// Assemble the load vector by iterating over all elements.
	b := mat.NewVecDense(n, nil)
	for i := 0; i < n-1; i++ {
		h := grid[i+1] - grid[i]
		b.SetVec(i, b.AtVec(i)+f(grid[i])*h/2)
		b.SetVec(i+1, b.AtVec(i+1)+f(grid[i+1])*h/2)
	}

	return System{A, b}
}

// Dense is a dense matrix
type Dense struct {
	D *mat.Dense
}

// Polynomial is a polynomial
type Polynomial struct {
	Bias    float64
	Weights [][]float64
}

// Matrix is a matrix of polynomials
type Matrix struct {
	Rows        int
	Cols        int
	Order       int
	Variables   int
	Polynomials [][]Polynomial
}

// NewMatrix generates a new matrix
func NewMatrix(rnd *rand.Rand, rows, cols, order, variables int) Matrix {
	polynomials := make([][]Polynomial, rows)
	for i := range polynomials {
		row := make([]Polynomial, cols)
		for j := range row {
			weights := make([][]float64, order)
			for k := range weights {
				weight := make([]float64, variables)
				for l := range weight {
					weight[l] = rnd.NormFloat64()
				}
				weights[k] = weight
			}
			row[j].Bias = rnd.NormFloat64()
			row[j].Weights = weights
		}
		polynomials[i] = row
	}
	return Matrix{
		Rows:        rows,
		Cols:        cols,
		Order:       order,
		Variables:   variables,
		Polynomials: polynomials,
	}
}

// ToMatrix converts a matrix to a matrix
func (m *Matrix) ToMatrix(parameters []float64) *Dense {
	data := make([]float64, 0, m.Rows*m.Cols)
	for _, row := range m.Polynomials {
		for _, col := range row {
			sum, prod := col.Bias, make([]float64, m.Variables)
			for i := range prod {
				prod[i] = 1
			}
			for _, weights := range col.Weights {
				for i, weight := range weights {
					prod[i] *= parameters[i]
					sum += prod[i] * weight
				}
			}
			data = append(data, sum)
		}
	}
	return &Dense{
		D: mat.NewDense(m.Rows, m.Cols, data),
	}
}

// ToVec converts a matrix to a vector
func (m *Matrix) ToVec(parameters []float64) *mat.VecDense {
	if m.Cols != 1 {
		panic(fmt.Errorf("cols should be 1 not %d", m.Cols))
	}
	data := make([]float64, 0, m.Rows)
	for _, row := range m.Polynomials {
		for _, col := range row {
			sum, prod := col.Bias, make([]float64, m.Variables)
			for i := range prod {
				prod[i] = 1
			}
			for _, weights := range col.Weights {
				for i, weight := range weights {
					prod[i] *= parameters[i]
					sum += prod[i] * weight
				}
			}
			data = append(data, sum)
		}
	}
	return mat.NewVecDense(m.Rows, data)
}

// MulVecTo computes A*x or Aᵀ*x and stores the result into dst.
func (d *Dense) MulVecTo(dst *mat.VecDense, trans bool, x mat.Vector) {
	if trans {
		dst.MulVec(d.D.T(), x)
	} else {
		dst.MulVec(d.D, x)
	}
}

func main() {
	const (
		n  = 10
		x0 = 0.0
		x1 = 1.0
	)
	// Make a uniform grid.
	grid := make([]float64, n+1)
	floats.Span(grid, x0, x1)
	sys := L2Projection(grid, func(x float64) float64 {
		return x * math.Sin(x)
	})

	result, err := linsolve.Iterative(sys.A, sys.B, &linsolve.CG{}, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Printf("# iterations: %v\n", result.Stats.Iterations)
	fmt.Printf("Final solution: %.6f\n", mat.Formatted(result.X.T()))

	// https://en.wikipedia.org/wiki/System_of_linear_equations
	a := &Dense{
		D: mat.NewDense(3, 3, []float64{3, 2, -1, 2, -2, 4, -1, .5, -1}),
	}
	b := mat.NewVecDense(3, []float64{1, -2, 0})

	result, err = linsolve.Iterative(a, b, &linsolve.GMRES{}, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	fmt.Printf("# iterations: %v\n", result.Stats.Iterations)
	fmt.Printf("Final solution: %.6f\n", mat.Formatted(result.X.T()))

	rnd := rand.New(rand.NewSource(1))
	aa := NewMatrix(rnd, 3, 3, 1, 4)
	bb := NewMatrix(rnd, 3, 1, 1, 4)
	inference := func(in []float64) []float64 {
		aaa := aa.ToMatrix(in)
		bbb := bb.ToVec(in)
		result, err = linsolve.Iterative(aaa, bbb, &linsolve.GMRES{}, nil)
		if err != nil {
			fmt.Println("Error:", err)
			return nil
		}

		//fmt.Printf("# iterations: %v\n", result.Stats.Iterations)
		return result.X.RawVector().Data
	}
	fmt.Printf("Final solution: %.6f\n", inference([]float64{1, 1, 1, 1}))

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	cost := func() float64 {
		fisher, cost := datum.Fisher, 0.0
		for _, value := range fisher {
			out := inference(value.Measures)
			target := make([]float64, 3)
			target[iris.Labels[value.Label]] = 1
			for i, value := range out {
				diff := target[i] - value
				cost += diff * diff
			}
		}
		return cost / 150
	}
	fmt.Println(cost())
}
