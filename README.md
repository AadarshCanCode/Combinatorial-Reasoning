# Combinatorial Reasoning (CR) with LLMs

This project explores the idea of coupling **combinatorial optimization** with **large language models (LLMs)** to improve reasoning.  
Inspired by the paper on Combinatorial Reasoning, the pipeline:
1. Samples multiple reasoning paths from an LLM.
2. Maps the reasoning selection problem into a **QUBO formulation**.
3. Uses specialized QUBO solvers (simulated annealing, D-Wave) to select the best subset of reasons.
4. Builds a Chain-of-Thought style prompt automatically.

## Features
- Automated reasoning path sampling from LLMs
- QUBO formulation + solver integration
- Baselines: Random selection, Majority rule
- Jupyter demo notebook

## Getting Started
```bash
git clone https://github.com/YOUR_USERNAME/combinatorial-reasoning.git
cd combinatorial-reasoning
pip install -r requirements.txt
