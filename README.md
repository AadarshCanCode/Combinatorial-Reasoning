### Interactive Web Demo (Recommended)

The easiest way to try CRQUBO is through our interactive Gradio web interface.

Note: the framework is modular — it supports multiple LLM and optimization backends. OpenAI is a common default, but you can configure other providers (local models, Hugging Face endpoints, Anthropic, or on-prem inference servers) via configuration or by implementing the simple backend adapter discussed below.

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) If you plan to use the OpenAI backend set your key; otherwise configure your preferred backend.
# Linux/macOS
export OPENAI_API_KEY="your-openai-api-key-here"
# Windows PowerShell
setx OPENAI_API_KEY "your-openai-api-key-here"

# Launch the web demo
python run_gradio_demo.py
```
### Run Examples

```bash
# Run built-in examples
python -m crqubo.main

# Launch interactive web demo
python run_gradio_demo.py

# Run Jupyter notebook examples
jupyter notebook examples/crqubo_demo.ipynb
```
The CRQUBO framework consists of eight modular components:

## 📚 Usage Examples

### Interactive Web Demo

The easiest way to explore CRQUBO is through our interactive Gradio web interface:

```bash
python run_gradio_demo.py
```

Features:
- 🎯 **One-click examples** for different reasoning domains
- ⚙️ **Real-time configuration** of pipeline settings
- 📊 **Performance analytics** and visualizations
- 📝 **Query history** with export functionality
- 🔄 **Live processing** with step-by-step reasoning display

### Causal Reasoning

```python
from crqubo import CRLLMPipeline

pipeline = CRLLMPipeline()

result = pipeline.process_query(
    query="How does education affect income?",
    domain="causal",
    use_retrieval=True
)

print(result.final_answer)
```
### Causal Reasoning

```python
from crqubo import CRLLMPipeline

pipeline = CRLLMPipeline()

result = pipeline.process_query(
    query="How does education affect income?",
    domain="causal",
    use_retrieval=True
)

print(result.final_answer)
```
```python
from crqubo.modules import TaskAgnosticInterface

interface = TaskAgnosticInterface()
processed = interface.process_input("Why does X cause Y?", domain="causal")
```
```python
from crqubo.modules import RetrievalModule

retrieval = RetrievalModule()
result = retrieval.retrieve("climate change causes", top_k=5)
```
```python
from crqubo.modules import ReasonSampler

sampler = ReasonSampler()
steps = sampler.sample_reasons("What causes inflation?", domain="causal")
```
```python
from crqubo.modules import SemanticFilter

filter_module = SemanticFilter()
filtered = filter_module.filter_reasons(reasoning_steps)
```
```python
from crqubo.modules import CombinatorialOptimizer

optimizer = CombinatorialOptimizer()
selected = optimizer.optimize_selection(reasoning_steps, query)
```
```python
from crqubo.modules import ReasonOrderer

orderer = ReasonOrderer()
```
# CRQUBO: Combinatorial Reasoning with Large Language Models

A modular reasoning framework that generalizes the Combinatorial Reasoning (CR) pipeline across diverse reasoning tasks using Large Language Models. The system combines zero-shot or few-shot LLM sampling, semantic deduplication, QUBO-based combinatorial selection, and optional knowledge retrieval (RAG) to construct optimal chains of reasoning for complex queries.

## 🚀 Key Features

- **Task-Agnostic Interface**: Accepts queries from any reasoning domain (causal, logical, spatial, arithmetic, etc.)
- **Optional Retrieval (RAG)**: Uses semantic search over external knowledge bases when needed
- **Reason Sampling**: Generates candidate reasoning steps using zero-shot or few-shot prompting
- **Semantic Filtering**: Removes near-duplicate reasoning chains using embeddings
- **Combinatorial Optimizer**: Solves QUBO problems to select diverse, high-utility reasoning steps
- **Reason Ordering**: Arranges selected reasons into logical chains (Chain-of-Thought or Tree-of-Thought)
- **Final Inference**: Generates coherent answers using the selected reasoning path
- **Reason Verifier**: Optional theorem prover integration (Z3) to filter inconsistent chains

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM functionality)
- Optional: D-Wave API key (for quantum optimization)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Combinatorial-Reasoning/crqubo.git
cd crqubo

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```
```bash
# Clone the repository
git clone https://github.com/AadarshCanCode/Combinatorial-Reasoning.git
cd Combinatorial-Reasoning

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Install Dependencies Only

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Interactive Web Demo (Recommended)

The easiest way to try CRQUBO is through our interactive Gradio web interface:

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Launch the web demo
python run_gradio_demo.py
```
```bash
# Set your OpenAI API key (Windows PowerShell example)
setx OPENAI_API_KEY "your-openai-api-key-here"

# Launch the web demo
python run_gradio_demo.py
```

The demo will open in your browser at `http://localhost:7860` with a user-friendly interface for testing different reasoning tasks.

### Basic Usage

```python
from crqubo import CRLLMPipeline

# Create a pipeline
pipeline = CRLLMPipeline()

# Process a query
result = pipeline.process_query(
    query="Why does smoking cause lung cancer?",
    domain="causal",
    use_retrieval=True,
    use_verification=True
)

print(f"Answer: {result.final_answer}")
print(f"Confidence: {result.confidence}")
print("Reasoning Chain:")
for i, step in enumerate(result.reasoning_chain, 1):
    print(f"{i}. {step}")
```

### Command Line Interface

```bash
# Basic usage
python -m crqubo.main "Why does smoking cause lung cancer?" --domain causal

# With retrieval and verification
python -m crqubo.main "What are the causes of climate change?" \
    --domain causal \
    --use-retrieval \
    --use-verification \
    --output result.json

# Using configuration file
python -m crqubo.main "Solve for x: 2x + 5 = 13" \
    --config config.json \
    --domain arithmetic
```
```bash
# Basic usage
python -m crqubo.main "Why does smoking cause lung cancer?" --domain causal

# With retrieval and verification
python -m crqubo.main "What are the causes of climate change?" \
    --domain causal \
    --use-retrieval \
    --use-verification \
    --output result.json

# Using configuration file
python -m crqubo.main "Solve for x: 2x + 5 = 13" \
    --config config.json \
    --domain arithmetic
```

### Run Examples

```bash
# Run built-in examples
python -m crqubo.main

# Launch interactive web demo
python run_gradio_demo.py

# Run Jupyter notebook examples
jupyter notebook examples/crllm_demo.ipynb
```
```bash
# Run built-in examples
python -m crqubo.main

# Launch interactive web demo
python run_gradio_demo.py

# Run Jupyter notebook examples
jupyter notebook examples/crllm_demo.ipynb
```

## 🏗 Architecture

The CRLLM framework consists of eight modular components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Task-Agnostic │    │   Optional      │    │   Reason        │
│   Input         │───▶│   Retrieval     │───▶│   Sampling      │
│   Interface     │    │   (RAG)         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Final         │◀───│   Reason        │◀───│   Semantic      │
│   Inference     │    │   Ordering      │    │   Filtering     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Combinatorial │
                       │   Optimizer     │
                       │   (QUBO)        │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Reason        │
                       │   Verifier      │
                       │   (Z3)          │
                       └─────────────────┘
```

### Module Descriptions

1. **Task-Agnostic Input Interface**: Normalizes inputs and detects reasoning domains
2. **Optional Retrieval (RAG)**: Retrieves relevant knowledge using semantic search
3. **Reason Sampling**: Generates candidate reasoning steps using LLMs
4. **Semantic Filtering**: Removes duplicates using embedding similarity
5. **Combinatorial Optimizer**: Selects optimal reasoning steps using QUBO optimization
6. **Reason Ordering**: Arranges steps into logical chains
7. **Final Inference**: Generates coherent final answers
8. **Reason Verifier**: Validates logical consistency using theorem provers

## 📚 Usage Examples

### Interactive Web Demo

The easiest way to explore CRLLM is through our interactive Gradio web interface:

```bash
python run_gradio_demo.py
```

Features:
- 🎯 **One-click examples** for different reasoning domains
- ⚙️ **Real-time configuration** of pipeline settings
- 📊 **Performance analytics** and visualizations
- 📝 **Query history** with export functionality
- 🔄 **Live processing** with step-by-step reasoning display

### Causal Reasoning

```python
from crllm import CRLLMPipeline

pipeline = CRLLMPipeline()

result = pipeline.process_query(
    query="How does education affect income?",
    domain="causal",
    use_retrieval=True
)

print(result.final_answer)
```

### Logical Reasoning

```python
result = pipeline.process_query(
    query="If A implies B and B implies C, what can we conclude about A and C?",
    domain="logical",
    use_verification=True
)
```

### Arithmetic Reasoning

```python
result = pipeline.process_query(
    query="What is 15% of 200?",
    domain="arithmetic"
)
```

### Multi-Domain Reasoning

```python
# The framework automatically detects domains
result = pipeline.process_query(
    query="Compare the advantages and disadvantages of renewable energy"
)
```

## ⚙️ Configuration

### Configuration File

Create a `config.json` file:

```json
{
  "use_retrieval": true,
  "use_verification": true,
  "reason_sampler": {
    "model": "gpt-3.5-turbo",
    "num_samples": 5,
    "temperature": 0.7
  },
  "semantic_filter": {
    "similarity_threshold": 0.8,
    "quality_threshold": 0.3
  },
  "combinatorial_optimizer": {
    "solver_type": "classical",
    "max_selections": 5
  }
}
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
export DWAVE_API_TOKEN="your-dwave-api-token"  # Optional
```

### Programmatic Configuration

```python
from crllm import CRLLMPipeline
from crllm.modules import ReasonSampler, SemanticFilter

# Custom configuration
config = {
    "reason_sampler": {
        "model": "gpt-4",
        "num_samples": 8,
        "temperature": 0.5
    },
    "semantic_filter": {
        "similarity_threshold": 0.75,
        "quality_threshold": 0.4
    }
}

pipeline = CRLLMPipeline(config=config)
```

## 🔧 API Reference

### CRLLMPipeline

Main pipeline class that orchestrates all modules.

#### Methods

- `process_query(query, domain=None, use_retrieval=False, use_verification=False, **kwargs)`: Process a single query
- `batch_process(queries, **kwargs)`: Process multiple queries
- `get_pipeline_info()`: Get pipeline configuration information

#### Parameters

- `query`: Input query (string or dict)
- `domain`: Reasoning domain ("causal", "logical", "arithmetic", etc.)
- `use_retrieval`: Enable knowledge retrieval
- `use_verification`: Enable reasoning verification

### Individual Modules

#### TaskAgnosticInterface

```python
from crllm.modules import TaskAgnosticInterface

interface = TaskAgnosticInterface()
processed = interface.process_input("Why does X cause Y?", domain="causal")
```

#### RetrievalModule

```python
from crllm.modules import RetrievalModule

retrieval = RetrievalModule()
result = retrieval.retrieve("climate change causes", top_k=5)
```

#### ReasonSampler

```python
from crllm.modules import ReasonSampler

sampler = ReasonSampler()
steps = sampler.sample_reasons("What causes inflation?", domain="causal")
```

#### SemanticFilter

```python
from crllm.modules import SemanticFilter

filter_module = SemanticFilter()
filtered = filter_module.filter_reasons(reasoning_steps)
```

#### CombinatorialOptimizer

```python
from crllm.modules import CombinatorialOptimizer

optimizer = CombinatorialOptimizer()
selected = optimizer.optimize_selection(reasoning_steps, query)
```

#### ReasonOrderer

```python
from crllm.modules import ReasonOrderer

orderer = ReasonOrderer()
ordered = orderer.order_reasons(reasoning_steps, query)
```

#### FinalInference

```python
from crllm.modules import FinalInference

inference = FinalInference()
result = inference.generate_answer(query, reasoning_chain)
```

#### ReasonVerifier

```python
from crllm.modules import ReasonVerifier

verifier = ReasonVerifier()
verified = verifier.verify_reasons(reasoning_steps, query)
```

## 🧪 Advanced Usage

### Custom Module Implementation

```python
from crllm.modules import BaseReasonSampler
from crllm import CRLLMPipeline

class CustomSampler(BaseReasonSampler):
    def sample_reasons(self, query, **kwargs):
        # Your custom implementation
        return reasoning_steps

# Use custom module
pipeline = CRLLMPipeline(reason_sampler=CustomSampler())
```

### Batch Processing

```python
queries = [
    "Why does smoking cause cancer?",
    "What is 20% of 150?",
    "How do vaccines work?"
]

results = pipeline.batch_process(queries, domain="general")
```

### Knowledge Base Setup

```python
from crllm.modules import RetrievalModule

retrieval = RetrievalModule()

# Add knowledge documents
documents = [
    {
        "content": "Smoking introduces harmful chemicals into the lungs...",
        "metadata": {"domain": "causal", "source": "medical_journal"}
    },
    # ... more documents
]

retrieval.add_knowledge(documents, domain="causal")
```

## 🔬 Domain-Specific Configurations

### Causal Reasoning

```python
causal_config = {
    "reason_sampler": {
        "strategy": "few_shot",
        "num_samples": 8,
        "temperature": 0.8
    },
    "semantic_filter": {
        "similarity_threshold": 0.75,
        "quality_threshold": 0.4
    },
    "combinatorial_optimizer": {
        "max_selections": 6,
        "diversity_weight": 0.6
    }
}
```

### Logical Reasoning

```python
logical_config = {
    "reason_sampler": {
        "strategy": "zero_shot",
        "temperature": 0.5
    },
    "reason_verifier": {
        "verification_level": "advanced",
        "check_contradictions": True
    }
}
```

### Arithmetic Reasoning

```python
arithmetic_config = {
    "reason_sampler": {
        "strategy": "few_shot",
        "temperature": 0.3
    },
    "combinatorial_optimizer": {
        "max_selections": 4,
        "utility_weight": 0.6
    }
}
```

## 🚀 Performance Optimization

### Solver Selection

```python
# Classical optimization (faster, good for most cases)
config = {
    "combinatorial_optimizer": {
        "solver_type": "classical"
    }
}

# Quantum optimization (slower, better for complex problems)
config = {
    "combinatorial_optimizer": {
        "solver_type": "quantum"
    }
}

# QAOA (quantum approximate optimization)
config = {
    "combinatorial_optimizer": {
        "solver_type": "qaoa"
    }
}
```

### Memory Optimization

```python
# Reduce memory usage
config = {
    "semantic_filter": {
        "model_name": "all-MiniLM-L6-v2"  # Smaller model
    },
    "reason_sampler": {
        "num_samples": 3  # Fewer samples
    }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Memory Issues with Large Models**
   - Use smaller embedding models
   - Reduce number of samples
   - Enable model caching

3. **Quantum Solver Not Available**
   - Falls back to classical solver automatically
   - Install D-Wave Ocean SDK for quantum optimization

4. **Z3 Verification Errors**
   - Install Z3: `pip install z3-solver`
   - Use basic verification level for simple cases

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose output
result = pipeline.process_query(
    query="test query",
    verbose=True
)
```

## 📊 Evaluation and Metrics

### Built-in Metrics

```python
result = pipeline.process_query(query)

# Access metrics
print(f"Confidence: {result.confidence}")
print(f"Reasoning steps: {len(result.reasoning_chain)}")
print(f"Used retrieval: {result.metadata['used_retrieval']}")
print(f"Used verification: {result.metadata['used_verification']}")
```

### Custom Evaluation

```python
def evaluate_reasoning_quality(result):
    # Your custom evaluation logic
    score = 0.0
    
    # Check answer quality
    if len(result.final_answer) > 50:
        score += 0.3
    
    # Check reasoning chain length
    if 3 <= len(result.reasoning_chain) <= 8:
        score += 0.3
    
    # Check confidence
    score += result.confidence * 0.4
    
    return score

quality_score = evaluate_reasoning_quality(result)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/crllm/crllm.git
cd crllm

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
flake8 crllm/
```

### Adding New Modules

1. Create a new module in `crllm/modules/`
2. Implement the base class interface
3. Add configuration options
4. Update the main pipeline
5. Add tests and documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Sentence Transformers for embeddings
- Z3 Theorem Prover for logical verification
- D-Wave for quantum optimization
- The broader AI research community

## 📞 Support

- **Documentation**: [https://crllm.readthedocs.io](https://crllm.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/CRLLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/CRLLM/discussions)
- **Email**: support@crllm.com

## 🔗 Related Projects

- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Original CoT paper
- [Tree-of-Thought Reasoning](https://arxiv.org/abs/2305.10601) - ToT reasoning framework
- [Combinatorial Optimization](https://en.wikipedia.org/wiki/Combinatorial_optimization) - QUBO optimization
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) - RAG methodology
- [Icosa](https://github.com/your-username/icosa) - Related reasoning framework
- [Combinatorial Reasoning Paper](https://arxiv.org/abs/your-paper-id) - Research paper

---

**CRLLM** - Advancing reasoning capabilities through combinatorial optimization and large language models.

