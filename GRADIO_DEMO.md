# CRLLM Gradio Demo

Interactive web interface for the CRLLM framework using Gradio. This demo provides a user-friendly way to test the reasoning capabilities across different domains.

## 🚀 Quick Start

### Prerequisites

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

### Running the Demo

#### Option 1: Using the launcher script (Recommended)
```bash
python run_gradio_demo.py
```

#### Option 2: Direct execution
```bash
python gradio_demo.py
```

#### Option 3: As a module
```bash
python -m gradio_demo
```

## 🌐 Interface Features

### Main Components

1. **Query Input Section**
   - Text input for your reasoning questions
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

### Interactive Features

- **Real-time Processing**: See results as they're generated
- **Domain Detection**: Automatic domain classification
- **Configuration Management**: Easy switching between different setups
- **Performance Tracking**: Monitor processing times and confidence scores
- **History Management**: Keep track of all queries and results
- **Data Export**: Export query history as JSON

## 📊 Example Queries

The demo includes pre-loaded examples for different reasoning domains:

### Causal Reasoning
- "Why does smoking cause lung cancer?"
- "What are the main causes of climate change?"

### Logical Reasoning
- "If all birds can fly and penguins are birds, can penguins fly?"
- "If A implies B and B implies C, what can we conclude about A and C?"

### Arithmetic Reasoning
- "What is 15% of 200?"
- "Solve for x: 2x + 5 = 13"

### General Reasoning
- "How can we improve team productivity?"
- "Compare renewable and fossil fuel energy sources"

## ⚙️ Configuration Options

### Basic Configuration
- **Domain**: Auto-detect or specify reasoning domain
- **Retrieval**: Enable/disable knowledge retrieval (RAG)
- **Verification**: Enable/disable reasoning verification

### Advanced Configuration
You can modify the `gradio_demo.py` file to customize:
- LLM model settings
- Sampling parameters
- Optimization strategies
- Filtering thresholds
- Ordering methods

## 📈 Performance Analytics

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

## 🔧 Customization

### Adding New Examples
Edit the `get_example_queries()` method in `CRLLMGradioDemo` class:

```python
def get_example_queries(self) -> List[List[str]]:
    return [
        ["Your new query", "domain", use_retrieval, use_verification],
        # ... more examples
    ]
```

### Modifying the Interface
The interface is built using Gradio blocks. You can customize:
- Layout and styling
- Input/output components
- Event handlers
- Visual themes

### Adding New Features
Extend the `CRLLMGradioDemo` class with new methods:
- Custom processing functions
- Additional visualizations
- Export formats
- Integration with external services

## 🐛 Troubleshooting

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
   Solution: Change the port in gradio_demo.py or kill the existing process
   ```

4. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce batch size or use smaller models
   ```

### Debug Mode

Enable debug mode by modifying the launch parameters:

```python
interface.launch(
    debug=True,
    show_error=True
)
```

## 🌐 Deployment

### Local Deployment
The demo runs locally by default on `http://localhost:7860`

### Cloud Deployment
For cloud deployment, modify the launch parameters:

```python
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Creates public link
    debug=False
)
```

### Docker Deployment
Create a Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "gradio_demo.py"]
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

## 🤝 Contributing

To contribute to the Gradio demo:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues with the Gradio demo:
- Check the troubleshooting section
- Review the console output
- Check the Gradio documentation
- Open an issue on GitHub

## 🎯 Future Enhancements

Planned features:
- Real-time collaboration
- Custom model integration
- Advanced visualizations
- Batch processing interface
- API endpoint integration
- User authentication
- Session management
- Custom themes
- Plugin system

---

**Enjoy exploring the CRLLM framework with the interactive Gradio demo!** 🚀
