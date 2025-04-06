# Metaheuristics Project: Eagle Strategy with Levy Flights and Differential Evolution

## Overview
The project combines two well-known metaheuristics for optimization:

1. **Levy Flight for Global Search**: A stochastic search strategy used for global exploration. It follows a Levy 
distribution to create a biased random walk.

2. **Differential Evolution for Local Search**: Once a promising region has been found, Differential Evolution is used 
as a local search method to refine solutions in that region. DE is applied iteratively, adjusting the population of 
agents with mutation, crossover, and selection strategies.

The **Eagle Strategy (ES)** first uses Levy Flights to explore the search space and then switches to Differential 
Evolution to fine-tune solutions in promising regions.

## Requirements
The following libraries are required to run the code:
- `numpy`
- `matplotlib`

To install the dependencies run:
```bash
pip install -r requirments.txt
```

## Usage
To run the code, execute the following command in your terminal:
```bash
python EagleStrategy.py
```