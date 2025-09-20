# CRQUBO Public Usage Guide

This short guide shows how to run the minimal examples and how to configure modular backends.

Run the example:

```powershell
python examples/simple_example.py
```

Modular backends

CRQUBO is backend-agnostic. Configure providers via `config.json` or environment variables.

Example `config.json`:

```json
{
  "llm_provider": "openai",
  "openai_api_key": "${OPENAI_API_KEY}",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "retriever": "chroma"
}
```

If you want to use local models or a different provider, set `llm_provider` to `local` or `huggingface` and implement the corresponding adapter. See the main repository for adapter examples.

Optional extras

Install Gradio for the interactive demo:

```powershell
pip install gradio
```

Optional backend dependencies (only if you use them):

- chromadb
- faiss-cpu
- sentence-transformers
- qiskit (for quantum optimizer backends)
- dwave-ocean-sdk (for D-Wave integrations)
