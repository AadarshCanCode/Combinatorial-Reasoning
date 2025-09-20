# CRLLM Demo Guide

This guide provides comprehensive instructions for running and using the various CRLLM demos.

## 🚀 Quick Start

### 1. Interactive Web Demo (Recommended)

The easiest way to explore CRLLM is through our interactive Gradio web interface:

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Launch the web demo
python run_gradio_demo.py
```

**Features:**
- 🎯 One-click examples for different reasoning domains
- ⚙️ Real-time configuration of pipeline settings
- 📊 Performance analytics and visualizations
- 📝 Query history with export functionality
- 🔄 Live processing with step-by-step reasoning display

### 2. Command Line Interface

For quick testing and automation:

```bash
# Basic usage
python -m crllm.main "Why does smoking cause lung cancer?" --domain causal

# With retrieval and verification
python -m crllm.main "What are the causes of climate change?" \
    --domain causal --use-retrieval --use-verification

# Using configuration file
python -m crllm.main "Solve for x: 2x + 5 = 13" \
    --config config.json --domain arithmetic
```

### 3. Jupyter Notebook

For interactive exploration and experimentation:

```bash
# Launch Jupyter notebook
jupyter notebook examples/crllm_demo.ipynb
```

### 4. Simple Python Demo

For basic testing without web interface:

```bash
python demo.py
```

## 🛠️ Installation Options

### Option 1: Automatic Installation

```bash
# Install dependencies and run demo
python launch_demo.py --install --demo gradio
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Run demo
python run_gradio_demo.py
```

### Option 3: Development Installation

```bash
# Install in development mode
pip install -e .

# Run tests
python test_gradio_demo.py

# Launch demo
python run_gradio_demo.py
```

## 🌐 Gradio Web Demo Features

### Main Interface

1. **Query Input Section**
   - Text input for reasoning questions
   - Domain selection (Auto-detect, Causal, Logical, Arithmetic, General)
   - Options for Knowledge Retrieval (RAG) and Reasoning Verification
   - Process and Clear buttons

2. **Configuration Panel**
   - Pipeline status display
   - Update and reset configuration options
   - Real-time configuration feedback

3. **Results Display**
   - Final answer output
   - Step-by-step reasoning chain
   - Processing metadata and statistics
   - Configuration information

4. **Example Queries**
   - Pre-loaded examples for different domains
   - One-click loading of example queries
   - Covers causal, logical, arithmetic, and general reasoning

5. **History & Analytics**
   - Query history table with all processed queries
   - Performance metrics and statistics
   - Interactive performance visualizations
   - Export functionality for history data

### Example Queries

The demo includes pre-loaded examples:

#### Causal Reasoning
- "Why does smoking cause lung cancer?"
- "What are the main causes of climate change?"

#### Logical Reasoning
- "If all birds can fly and penguins are birds, can penguins fly?"
- "If A implies B and B implies C, what can we conclude about A and C?"

#### Arithmetic Reasoning
- "What is 15% of 200?"
- "Solve for x: 2x + 5 = 13"

#### General Reasoning
- "How can we improve team productivity?"
- "Compare renewable and fossil fuel energy sources"

## ⚙️ Configuration Options

### Basic Configuration
- **Domain**: Auto-detect or specify reasoning domain
- **Retrieval**: Enable/disable knowledge retrieval (RAG)
- **Verification**: Enable/disable reasoning verification

### Advanced Configuration
Modify `gradio_config.json` or edit `gradio_demo.py` to customize:
- LLM model settings
- Sampling parameters
- Optimization strategies
- Filtering thresholds
- Ordering methods

## 📊 Performance Analytics

The demo includes built-in analytics:

1. **Processing Time Analysis**
   - Average processing time by domain
   - Time vs confidence correlation
   - Performance trends over time

2. **Confidence Metrics**
   - Confidence score distribution
   - Domain-specific confidence patterns
   - Quality assessment

3. **Reasoning Chain Analysis**
   - Average reasoning steps by domain
   - Chain complexity metrics
   - Step quality indicators

## 🔧 Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: OpenAI API key not found
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Missing Dependencies**
   ```
   Error: No module named 'gradio'
   Solution: pip install -r requirements.txt
   ```

3. **Port Already in Use**
   ```
   Error: Port 7860 is already in use
   Solution: Change the port or kill existing process
   ```

4. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce batch size or use smaller models
   ```

### Debug Mode

Enable debug mode:

```bash
python launch_demo.py --demo gradio --debug
```

### Test Suite

Run the test suite to verify everything works:

```bash
python test_gradio_demo.py
```

## 🚀 Advanced Usage

### Custom Configuration

Create a custom configuration file:

```json
{
  "use_retrieval": true,
  "use_verification": true,
  "reason_sampler": {
    "num_samples": 8,
    "temperature": 0.7
  },
  "combinatorial_optimizer": {
    "solver_type": "quantum"
  }
}
```

### Batch Processing

Process multiple queries programmatically:

```python
from crllm import CRLLMPipeline

pipeline = CRLLMPipeline()
queries = [
    "Why does smoking cause cancer?",
    "What is 20% of 150?",
    "How do vaccines work?"
]

results = pipeline.batch_process(queries, domain="general")
```

### Custom Modules

Extend the framework with custom modules:

```python
from crllm.modules import BaseReasonSampler

class CustomSampler(BaseReasonSampler):
    def sample_reasons(self, query, **kwargs):
        # Your custom implementation
        return reasoning_steps

pipeline = CRLLMPipeline(reason_sampler=CustomSampler())
```

## 📱 Mobile Support

The Gradio interface is responsive and works on mobile devices:
- Touch-friendly interface
- Responsive layout
- Mobile-optimized components

## 🔒 Security Considerations

- API keys are handled securely
- No data is stored permanently (except in browser)
- All processing happens locally
- History can be cleared at any time

## 📊 Usage Statistics

The demo tracks:
- Number of queries processed
- Average processing time
- Domain distribution
- Success/failure rates
- User interaction patterns

## 🎯 Best Practices

1. **Start Simple**: Begin with basic queries and gradually increase complexity
2. **Use Examples**: Try the pre-loaded examples to understand different domains
3. **Monitor Performance**: Use the analytics to understand processing patterns
4. **Experiment with Configuration**: Try different settings to optimize for your use case
5. **Export Results**: Use the export functionality to save interesting results

## 🤝 Contributing

To contribute to the demos:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues with the demos:
- Check the troubleshooting section
- Review the console output
- Check the documentation
- Open an issue on GitHub

## 🎉 Enjoy Exploring CRLLM!

The demos provide a comprehensive way to explore the CRLLM framework's capabilities. Start with the Gradio web demo for the best interactive experience, then explore the other demos to understand different use cases and integration patterns.

---

**Happy reasoning with CRLLM!** 🚀
