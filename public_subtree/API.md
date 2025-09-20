# CRQUBO Public API (minimal)

This document gives a minimal overview of the public-facing API exposed by the `crqubo` package. For full developer documentation, see the main repository.

Primary class

- `CRLLMPipeline(config: dict | None = None)`
  - Create a reasoning pipeline. Pass a configuration dictionary to customize providers, retrievers, and verification.
  - Key methods:
    - `process_query(query: str, domain: Optional[str], use_retrieval: bool, use_verification: bool)` — run a query through the pipeline and obtain a result object containing `final_answer`, `confidence`, `reasoning_chain`, and `metadata`.

Result object (summary)

- `final_answer: str`
- `confidence: float`
- `reasoning_chain: List[str]`
- `metadata: Dict[str, Any]`

Adapters and modular backends

- The package is designed around provider adapters. Configure `llm_provider` and adapter settings in `config.json` or by passing a `config` dict to `CRLLMPipeline`.
- Example providers: `openai`, `local`, `huggingface`. Each provider may require additional configuration (API keys, model names).

Example usage

```python
from crqubo import CRLLMPipeline
pipeline = CRLLMPipeline(config={"llm_provider": "openai"})
result = pipeline.process_query("Why does smoking cause lung cancer?", domain="causal", use_retrieval=False, use_verification=False)
print(result.final_answer)
```
