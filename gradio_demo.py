#!/usr/bin/env python3
"""
CRLLM Gradio Demo

Interactive web interface for the CRLLM framework using Gradio.
This demo allows users to test different reasoning tasks and configurations.
"""

import os
import sys
import time
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from crllm import CRLLMPipeline
from crllm.modules import (
    TaskAgnosticInterface,
    RetrievalModule,
    ReasonSampler,
    SemanticFilter,
    CombinatorialOptimizer,
    ReasonOrderer,
    FinalInference,
    ReasonVerifier
)


class CRLLMGradioDemo:
    """Gradio demo class for CRLLM framework."""
    
    def __init__(self):
        """Initialize the demo."""
        self.pipeline = None
        self.current_config = {}
        self.history = []
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Initialize with default configuration
        self.update_pipeline({})
    
    def update_pipeline(self, config: Dict[str, Any]) -> str:
        """Update the pipeline with new configuration."""
        try:
            self.current_config = config
            self.pipeline = CRLLMPipeline(config=config)
            return "✅ Pipeline updated successfully!"
        except Exception as e:
            return f"❌ Error updating pipeline: {str(e)}"
    
    def process_query(
        self,
        query: str,
        domain: str,
        use_retrieval: bool,
        use_verification: bool,
        show_reasoning: bool = True
    ) -> Tuple[str, str, str, str, str]:
        """Process a query through the CRLLM pipeline."""
        if not query.strip():
            return "Please enter a query.", "", "", "", ""
        
        try:
            start_time = time.time()
            
            result = self.pipeline.process_query(
                query=query,
                domain=domain if domain != "auto" else None,
                use_retrieval=use_retrieval,
                use_verification=use_verification
            )
            
            processing_time = time.time() - start_time
            
            # Store in history
            self.history.append({
                'query': query,
                'domain': result.metadata.get('domain', 'unknown'),
                'answer': result.final_answer,
                'confidence': result.confidence,
                'processing_time': processing_time,
                'reasoning_steps': len(result.reasoning_chain),
                'used_retrieval': result.metadata.get('used_retrieval', False),
                'used_verification': result.metadata.get('used_verification', False)
            })
            
            # Format reasoning chain
            reasoning_text = ""
            if show_reasoning and result.reasoning_chain:
                reasoning_text = "Reasoning Chain:\n"
                for i, step in enumerate(result.reasoning_chain, 1):
                    reasoning_text += f"{i}. {step}\n"
            
            # Format metadata
            metadata_text = f"""
Processing Time: {processing_time:.2f}s
Confidence: {result.confidence:.2f}
Domain: {result.metadata.get('domain', 'unknown')}
Reasoning Steps: {len(result.reasoning_chain)}
Used Retrieval: {'Yes' if result.metadata.get('used_retrieval') else 'No'}
Used Verification: {'Yes' if result.metadata.get('used_verification') else 'No'}
            """.strip()
            
            # Format configuration info
            config_info = f"""
Current Configuration:
- Retrieval: {'Enabled' if use_retrieval else 'Disabled'}
- Verification: {'Enabled' if use_verification else 'Disabled'}
- Domain: {domain}
- Pipeline Modules: {len(self.pipeline.get_pipeline_info()['modules'])}
            """.strip()
            
            return (
                result.final_answer,
                reasoning_text,
                metadata_text,
                config_info,
                f"✅ Query processed successfully in {processing_time:.2f}s"
            )
            
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            return error_msg, "", "", "", error_msg
    
    def get_example_queries(self) -> List[List[str]]:
        """Get example queries for different domains."""
        return [
            ["Why does smoking cause lung cancer?", "causal", True, True],
            ["If all birds can fly and penguins are birds, can penguins fly?", "logical", False, True],
            ["What is 15% of 200?", "arithmetic", False, False],
            ["How can we improve team productivity?", "general", True, False],
            ["What are the main causes of climate change?", "causal", True, True],
            ["Solve for x: 2x + 5 = 13", "arithmetic", False, False],
            ["Compare renewable and fossil fuel energy sources", "general", True, False],
            ["If A implies B and B implies C, what can we conclude about A and C?", "logical", False, True]
        ]
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """Get history as a pandas DataFrame."""
        if not self.history:
            return pd.DataFrame(columns=[
                'Query', 'Domain', 'Confidence', 'Processing Time (s)', 
                'Reasoning Steps', 'Used Retrieval', 'Used Verification'
            ])
        
        df = pd.DataFrame(self.history)
        return df[['query', 'domain', 'confidence', 'processing_time', 
                  'reasoning_steps', 'used_retrieval', 'used_verification']].rename(columns={
            'query': 'Query',
            'domain': 'Domain', 
            'confidence': 'Confidence',
            'processing_time': 'Processing Time (s)',
            'reasoning_steps': 'Reasoning Steps',
            'used_retrieval': 'Used Retrieval',
            'used_verification': 'Used Verification'
        })
    
    def create_performance_plot(self) -> str:
        """Create a performance visualization."""
        if len(self.history) < 2:
            return None
        
        df = pd.DataFrame(self.history)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('CRLLM Performance Analysis', fontsize=16)
        
        # Processing time by domain
        domain_times = df.groupby('domain')['processing_time'].mean()
        axes[0, 0].bar(domain_times.index, domain_times.values, color='skyblue')
        axes[0, 0].set_title('Average Processing Time by Domain')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence distribution
        axes[0, 1].hist(df['confidence'], bins=10, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        
        # Processing time vs confidence
        scatter = axes[1, 0].scatter(df['processing_time'], df['confidence'], 
                                   c=df['reasoning_steps'], cmap='viridis', alpha=0.7)
        axes[1, 0].set_title('Processing Time vs Confidence')
        axes[1, 0].set_xlabel('Processing Time (s)')
        axes[1, 0].set_ylabel('Confidence')
        plt.colorbar(scatter, ax=axes[1, 0], label='Reasoning Steps')
        
        # Reasoning steps by domain
        domain_steps = df.groupby('domain')['reasoning_steps'].mean()
        axes[1, 1].bar(domain_steps.index, domain_steps.values, color='orange')
        axes[1, 1].set_title('Average Reasoning Steps by Domain')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot to temporary file
        plot_path = "temp_performance_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def clear_history(self) -> Tuple[str, str, str, str, str, str]:
        """Clear the query history."""
        self.history = []
        return "", "", "", "", "", "✅ History cleared successfully!"
    
    def export_history(self) -> str:
        """Export history as JSON."""
        if not self.history:
            return "No history to export."
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"crllm_history_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        
        return f"✅ History exported to {filename}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="CRLLM Demo",
            theme=gr.themes.Soft(),
            css="""
            .main-container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 2rem; }
            .section { margin: 1rem 0; }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🚀 CRLLM: Combinatorial Reasoning with Large Language Models
            
            Interactive demo of the CRLLM framework for advanced reasoning tasks across multiple domains.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Query Input Section
                    with gr.Group():
                        gr.Markdown("### 📝 Query Input")
                        
                        query_input = gr.Textbox(
                            label="Enter your query",
                            placeholder="e.g., Why does smoking cause lung cancer?",
                            lines=3
                        )
                        
                        with gr.Row():
                            domain_dropdown = gr.Dropdown(
                                choices=[
                                    ("Auto-detect", "auto"),
                                    ("Causal", "causal"),
                                    ("Logical", "logical"),
                                    ("Arithmetic", "arithmetic"),
                                    ("General", "general")
                                ],
                                value="auto",
                                label="Reasoning Domain"
                            )
                            
                            with gr.Column():
                                retrieval_checkbox = gr.Checkbox(
                                    label="Use Knowledge Retrieval (RAG)",
                                    value=False
                                )
                                verification_checkbox = gr.Checkbox(
                                    label="Use Reasoning Verification",
                                    value=False
                                )
                        
                        with gr.Row():
                            process_btn = gr.Button("🚀 Process Query", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                with gr.Column(scale=1):
                    # Configuration Section
                    with gr.Group():
                        gr.Markdown("### ⚙️ Configuration")
                        
                        config_status = gr.Textbox(
                            label="Pipeline Status",
                            value="✅ Pipeline ready",
                            interactive=False
                        )
                        
                        with gr.Row():
                            update_config_btn = gr.Button("Update Pipeline", size="sm")
                            reset_config_btn = gr.Button("Reset to Default", size="sm")
            
            # Results Section
            with gr.Group():
                gr.Markdown("### 📊 Results")
                
                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="Final Answer",
                            lines=4,
                            interactive=False
                        )
                        
                        reasoning_output = gr.Textbox(
                            label="Reasoning Chain",
                            lines=6,
                            interactive=False
                        )
                    
                    with gr.Column():
                        metadata_output = gr.Textbox(
                            label="Metadata",
                            lines=4,
                            interactive=False
                        )
                        
                        config_output = gr.Textbox(
                            label="Configuration Info",
                            lines=4,
                            interactive=False
                        )
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
            
            # Examples Section
            with gr.Group():
                gr.Markdown("### 💡 Example Queries")
                
                examples = gr.Examples(
                    examples=self.get_example_queries(),
                    inputs=[query_input, domain_dropdown, retrieval_checkbox, verification_checkbox],
                    label="Click to load example"
                )
            
            # History and Analytics Section
            with gr.Group():
                gr.Markdown("### 📈 History & Analytics")
                
                with gr.Row():
                    with gr.Column():
                        history_table = gr.Dataframe(
                            label="Query History",
                            interactive=False,
                            wrap=True
                        )
                        
                        with gr.Row():
                            refresh_history_btn = gr.Button("🔄 Refresh History")
                            clear_history_btn = gr.Button("🗑️ Clear History")
                            export_history_btn = gr.Button("📤 Export History")
                    
                    with gr.Column():
                        performance_plot = gr.Image(
                            label="Performance Analysis",
                            visible=False
                        )
                        
                        plot_btn = gr.Button("📊 Generate Performance Plot")
            
            # Event Handlers
            process_btn.click(
                fn=self.process_query,
                inputs=[query_input, domain_dropdown, retrieval_checkbox, verification_checkbox],
                outputs=[answer_output, reasoning_output, metadata_output, config_output, status_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", "", "", "", "✅ Cleared successfully!"),
                outputs=[query_input, answer_output, reasoning_output, metadata_output, config_output, status_output]
            )
            
            update_config_btn.click(
                fn=self.update_pipeline,
                inputs=gr.State({}),
                outputs=config_status
            )
            
            reset_config_btn.click(
                fn=lambda: self.update_pipeline({}),
                outputs=config_status
            )
            
            refresh_history_btn.click(
                fn=self.get_history_dataframe,
                outputs=history_table
            )
            
            clear_history_btn.click(
                fn=self.clear_history,
                outputs=[query_input, answer_output, reasoning_output, metadata_output, config_output, status_output]
            )
            
            export_history_btn.click(
                fn=self.export_history,
                outputs=status_output
            )
            
            plot_btn.click(
                fn=self.create_performance_plot,
                outputs=performance_plot
            ).then(
                fn=lambda x: gr.update(visible=x is not None),
                inputs=performance_plot,
                outputs=performance_plot
            )
            
            # Auto-refresh history when processing
            process_btn.click(
                fn=self.get_history_dataframe,
                outputs=history_table
            )
        
        return interface


def main():
    """Run the Gradio demo."""
    print("🚀 Starting CRLLM Gradio Demo...")
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
    
    # Create demo instance
    demo = CRLLMGradioDemo()
    
    # Create and launch interface
    interface = demo.create_interface()
    
    print("🌐 Launching Gradio interface...")
    print("   The demo will open in your browser automatically.")
    print("   If it doesn't open, check the terminal for the local URL.")
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
