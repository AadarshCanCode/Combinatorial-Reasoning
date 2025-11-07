# CRQUBO Demo Guide

This guide explains how to explore CRQUBO without the legacy Gradio interface. The web UI is being redesigned; see [`UI plan.md`](UI%20plan.md) for the upcoming experience. For now, use the CLI, notebooks, or lightweight Python scripts outlined below.

## 🚀 Quick Start

### 1. Command Line Interface (Recommended)
Run the full reasoning pipeline directly from the terminal.

```bash
# Basic usage
python -m crqubo.main "Why does smoking cause lung cancer?" --domain causal

# With retrieval and verification
python -m crqubo.main "What are the causes of climate change?" \
   --domain causal --use-retrieval --use-verification

# Using a configuration file
python -m crqubo.main "Solve for x: 2x + 5 = 13" \
   --config config.json --domain arithmetic
```

### 2. Jupyter Notebook Exploration
Leverage notebooks for interactive experimentation and visualization.

```bash
# Launch the example notebook
jupyter notebook examples/crqubo_demo.ipynb
```

### 3. Minimal Python Demo
The repository includes a simple script for quick smoke testing of the pipeline.

```bash
python demo.py
```

### 4. Upcoming Browser UI
A modern web interface is under active development. Until it ships, the recommended paths are the CLI and notebook workflows. Track progress in [`UI plan.md`](UI%20plan.md).

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Configure API keys
export OPENAI_API_KEY="your-openai-api-key-here"
export DWAVE_API_TOKEN="your-dwave-api-token"   # Optional for quantum solvers

# Install the package in editable mode (useful for development)
pip install -e .
```

## ✅ Verifying Your Setup

```bash
# Run the unit test suite
pytest

# Execute the pipeline on a sample question
python -m crqubo.main "How does education affect income?" --domain causal
```

## 🔁 Cleaning Up Legacy Assets

All Gradio-specific entry points have been removed. If you previously created local shortcuts or scripts that reference `run_gradio_demo.py`, please delete them. The new UI will introduce a replacement workflow.

## 📬 Need Help?

- Review the main [README](README.md) for architecture details.
- File issues or questions in the GitHub tracker.
- Reach out to the maintainers if you want to contribute to the new UI effort.
