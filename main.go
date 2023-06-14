// Copyright 2023 The Meta Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/datum/iris"

	"gonum.org/v1/exp/linsolve"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

const (
	//TrainSize is the size of the training set
	TrainSize = 2
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

// Polynomial is a polynomial copy
func (p *Polynomial) Copy() Polynomial {
	q := Polynomial{
		Bias:    p.Bias,
		Weights: make([][]float64, 0, len(p.Weights)),
	}
	for _, w := range p.Weights {
		weights := make([]float64, len(w))
		copy(weights, w)
		q.Weights = append(q.Weights, weights)
	}
	return q
}

// Matrix is a matrix of polynomials
type Matrix struct {
	Rows        int
	Cols        int
	Order       int
	Variables   int
	Polynomials [][]Polynomial
}

// Copy makes a copy of matrix
func (m *Matrix) Copy() Matrix {
	n := Matrix{
		Rows:        m.Rows,
		Cols:        m.Cols,
		Order:       m.Order,
		Variables:   m.Variables,
		Polynomials: make([][]Polynomial, len(m.Polynomials)),
	}
	for i, p := range m.Polynomials {
		polynomials := make([]Polynomial, len(p))
		for j := range polynomials {
			polynomials[j] = m.Polynomials[i][j].Copy()
		}
		n.Polynomials[i] = polynomials
	}
	return n
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

// Pair is a pair of matrices
type Pair struct {
	Clean bool
	Cost  float64
	A     Matrix
	B     Matrix
}

// Copy copies a pair
func (p *Pair) Copy() Pair {
	q := Pair{
		Clean: false,
		Cost:  p.Cost,
		A:     p.A.Copy(),
		B:     p.B.Copy(),
	}
	return q
}

// Inference does inference on the pair
func (p *Pair) Inference(in []float64) []float64 {
	a := p.A.ToMatrix(in)
	b := p.B.ToVec(in)
	result, err := linsolve.Iterative(a, b, &linsolve.GMRES{}, nil)
	if err != nil {
		fmt.Println("Error:", err)
		return nil
	}

	//fmt.Printf("# iterations: %v\n", result.Stats.Iterations)
	return result.X.RawVector().Data
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
	pair := Pair{
		A: NewMatrix(rnd, 3, 3, 1, 4),
		B: NewMatrix(rnd, 3, 1, 1, 4),
	}

	fmt.Printf("Final solution: %.6f\n", pair.Inference([]float64{1, 1, 1, 1}))

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}

	pairs := make([]Pair, 100)
	for i := range pairs {
		pairs[i].A = NewMatrix(rnd, 3, 3, 2, 4)
		pairs[i].B = NewMatrix(rnd, 3, 1, 2, 4)
	}
	train := make([]iris.Iris, 0, 150)
	for i := 0; i < 150; i += 50 {
		train = append(train, datum.Fisher[i:i+TrainSize]...)
	}
	cost := func(pair Pair) float64 {
		fisher, cost := train, 0.0
		for _, value := range fisher {
			out := pair.Inference(value.Measures)
			target := make([]float64, 3)
			target[iris.Labels[value.Label]] = 1
			for i, value := range out {
				diff := target[i] - value
				cost += diff * diff
			}
		}
		return cost / 150
	}
	fmt.Println(cost(pair))

	for e := 0; e < 1024; e++ {
		for i := range pairs {
			if pairs[i].Clean {
				continue
			}
			pairs[i].Cost = cost(pairs[i])
			pairs[i].Clean = true
		}
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].Cost < pairs[j].Cost
		})
		pairs = pairs[:100]
		fmt.Println(e, pairs[0].Cost)
		for j := 0; j < 100; j++ {
			if rnd.Intn(2) == 0 {
				if rnd.Intn(2) == 0 {
					if rnd.Intn(4) == 0 {
						a, b := rnd.Intn(20), rnd.Intn(20)
						arow := rnd.Intn(pairs[a].A.Rows)
						acol := rnd.Intn(pairs[a].A.Cols)
						brow := rnd.Intn(pairs[b].A.Rows)
						bcol := rnd.Intn(pairs[b].A.Cols)
						x := pairs[a].A.Polynomials[arow][acol].Bias
						y := pairs[b].A.Polynomials[brow][bcol].Bias
						xx := pairs[a].Copy()
						yy := pairs[b].Copy()
						xx.A.Polynomials[arow][acol].Bias = y
						yy.A.Polynomials[brow][bcol].Bias = x
						pairs = append(pairs, xx)
						pairs = append(pairs, yy)
					} else {
						a, b := rnd.Intn(20), rnd.Intn(20)
						arow := rnd.Intn(pairs[a].A.Rows)
						acol := rnd.Intn(pairs[a].A.Cols)
						aorder := rnd.Intn(pairs[a].A.Order)
						avars := rnd.Intn(pairs[a].A.Variables)
						brow := rnd.Intn(pairs[b].A.Rows)
						bcol := rnd.Intn(pairs[b].A.Cols)
						border := rnd.Intn(pairs[b].A.Order)
						bvars := rnd.Intn(pairs[b].A.Variables)
						xx := pairs[a].Copy()
						yy := pairs[b].Copy()
						x := xx.A.Polynomials[arow][acol].Weights[aorder][avars]
						y := yy.A.Polynomials[brow][bcol].Weights[border][bvars]
						xx.A.Polynomials[arow][acol].Weights[aorder][avars] = y
						yy.A.Polynomials[brow][bcol].Weights[border][bvars] = x
						pairs = append(pairs, xx)
						pairs = append(pairs, yy)
					}
				} else {
					if rnd.Intn(4) == 0 {
						a, b := rnd.Intn(20), rnd.Intn(20)
						arow := rnd.Intn(pairs[a].B.Rows)
						acol := rnd.Intn(pairs[a].B.Cols)
						brow := rnd.Intn(pairs[b].B.Rows)
						bcol := rnd.Intn(pairs[b].B.Cols)
						x := pairs[a].B.Polynomials[arow][acol].Bias
						y := pairs[b].B.Polynomials[brow][bcol].Bias
						xx := pairs[a].Copy()
						yy := pairs[b].Copy()
						xx.B.Polynomials[arow][acol].Bias = y
						yy.B.Polynomials[brow][bcol].Bias = x
						pairs = append(pairs, xx)
						pairs = append(pairs, yy)
					} else {
						a, b := rnd.Intn(20), rnd.Intn(20)
						arow := rnd.Intn(pairs[a].B.Rows)
						acol := rnd.Intn(pairs[a].B.Cols)
						aorder := rnd.Intn(pairs[a].B.Order)
						avars := rnd.Intn(pairs[a].B.Variables)
						brow := rnd.Intn(pairs[b].B.Rows)
						bcol := rnd.Intn(pairs[b].B.Cols)
						border := rnd.Intn(pairs[b].B.Order)
						bvars := rnd.Intn(pairs[b].B.Variables)
						xx := pairs[a].Copy()
						yy := pairs[b].Copy()
						x := xx.B.Polynomials[arow][acol].Weights[aorder][avars]
						y := yy.B.Polynomials[brow][bcol].Weights[border][bvars]
						xx.B.Polynomials[arow][acol].Weights[aorder][avars] = y
						yy.B.Polynomials[brow][bcol].Weights[border][bvars] = x
						pairs = append(pairs, xx)
						pairs = append(pairs, yy)
					}
				}
			} else {
				if rnd.Intn(2) == 0 {
					a := rnd.Intn(100)
					arow := rnd.Intn(pairs[a].A.Rows)
					acol := rnd.Intn(pairs[a].A.Cols)
					aorder := rnd.Intn(pairs[a].A.Order)
					avars := rnd.Intn(pairs[a].A.Variables)
					xx := pairs[a].Copy()
					x := xx.A.Polynomials[arow][acol].Weights[aorder][avars]
					xx.A.Polynomials[arow][acol].Weights[aorder][avars] = x + rnd.ExpFloat64()/10
					pairs = append(pairs, xx)
				} else {
					a := rnd.Intn(100)
					arow := rnd.Intn(pairs[a].B.Rows)
					acol := rnd.Intn(pairs[a].B.Cols)
					aorder := rnd.Intn(pairs[a].B.Order)
					avars := rnd.Intn(pairs[a].B.Variables)
					xx := pairs[a].Copy()
					x := xx.B.Polynomials[arow][acol].Weights[aorder][avars]
					xx.B.Polynomials[arow][acol].Weights[aorder][avars] = x + rnd.ExpFloat64()/10
					pairs = append(pairs, xx)
				}
			}
		}
	}

	correct := func(pair Pair) int {
		fisher, correct := datum.Fisher, 0
		for _, value := range fisher {
			out := pair.Inference(value.Measures)
			max, index := 0.0, 0
			for i, value := range out {
				if value > max {
					max, index = value, i
				}
			}
			if index == iris.Labels[value.Label] {
				correct++
			}
		}
		return correct
	}
	metaCorrect := correct(pairs[0])

	nearestNeighborCorrect := 0
	for _, value := range datum.Fisher {
		min, index := math.MaxFloat64, 0
		for i, neighbor := range train {
			distance := 0.0
			for j, measure := range neighbor.Measures {
				diff := measure - value.Measures[j]
				distance += diff * diff
			}
			if distance < min {
				min, index = distance, i
			}
		}
		if index == iris.Labels[value.Label] {
			nearestNeighborCorrect++
		}
	}

	fmt.Println("meta correct=", metaCorrect, float64(metaCorrect)/float64(len(datum.Fisher)))
	fmt.Println("nearest neighbor correct=", nearestNeighborCorrect, float64(nearestNeighborCorrect)/float64(len(datum.Fisher)))
}
