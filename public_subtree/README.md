# CRQUBO — Public Subtree

This public subtree contains a minimal, publishable slice of the CRQUBO project. It is intended for users who want a lightweight way to try the library and the basic example without cloning the full repository.

What is included

- `USAGE.md` — quick usage guide and modular backend examples
- `examples/` — minimal runnable example(s)
- `requirements-extras.txt` — optional dependencies for demos and backends
- `LICENSE` — MIT license
- `PUBLISHING.md` — how to publish this subtree
- `package_info.txt` — short package metadata summary

Quick start

1. Clone this repository (or install package if published)

```powershell
git clone https://github.com/AadarshCanCode/CombinatorialReasoning.git
cd Combinatorial-Reasoning
```

2. Optionally create a virtual environment and install extras for running the demo:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r public_subtree/requirements-extras.txt
```

3. Run the public example:

```powershell
python public_subtree/examples/simple_example.py
```

Notes on backends and API keys

- CRQUBO is backend-agnostic. OpenAI is an optional provider; you do not need an OpenAI API key to run the minimal example if you configure a local adapter.
- See `USAGE.md` for configuration examples and how to provide adapters.

Contributing

This subtree is a small, public-friendly snapshot. Contributions and issues should be opened in the main repository: https://github.com/AadarshCanCode/CombinatorialReasoning
