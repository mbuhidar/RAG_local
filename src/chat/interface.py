"""
Chat Interface Module

Provides Gradio-based chat interface for the RAG system.
"""

import logging
from typing import Dict, List, Optional

import gradio as gr

from ..rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ChatInterface:
    """Gradio chat interface for RAG system."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        title: str = "Local RAG Chat",
        description: str = "Chat with your documents using local LLM"
    ):
        """
        Initialize chat interface.

        Args:
            rag_pipeline: RAG pipeline instance
            title: Interface title
            description: Interface description
        """
        self.rag_pipeline = rag_pipeline
        self.title = title
        self.description = description
        self.chat_history: List[Dict[str, str]] = []

    def process_message(
        self,
        message: str,
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Process user message and return response.

        Args:
            message: User message
            history: Chat history (list of message dicts with 'role' and 'content')

        Returns:
            Updated history
        """
        if not message.strip():
            return history

        logger.info(f"Processing message: {message}")

        try:
            # Query RAG system
            result = self.rag_pipeline.query(
                question=message,
                return_sources=True,
                stream=False
            )

            response = result["answer"]
            
            # Add sources if available
            if "sources" in result and result["sources"]:
                sources_text = "\n\n**Sources:**\n"
                for i, source in enumerate(result["sources"][:3], 1):
                    filename = source["metadata"].get("filename", "Unknown")
                    score = source["score"]
                    sources_text += f"{i}. {filename} (relevance: {score:.2f})\n"
                response += sources_text

            # Update history with new message format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            self.chat_history = history

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            response = f"Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

        return history

    def clear_history(self) -> List[Dict[str, str]]:
        """Clear chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")
        return []

    def get_stats(self) -> str:
        """Get system statistics."""
        try:
            stats = self.rag_pipeline.get_stats()
            
            output = "**System Statistics**\n\n"
            output += f"**Vector Store:**\n"
            output += f"- Collection: {stats['vector_store']['name']}\n"
            output += f"- Documents: {stats['vector_store']['count']}\n\n"
            
            output += f"**Embedding Model:**\n"
            output += f"- Model: {stats['embedding_model']['model_name']}\n"
            output += f"- Dimensions: {stats['embedding_model']['embedding_dimension']}\n"
            output += f"- Device: {stats['embedding_model']['device']}\n\n"
            
            output += f"**LLM:**\n"
            output += f"- Provider: {stats['llm']['provider']}\n"
            output += f"- Model: {stats['llm']['model_name']}\n"
            output += f"- Temperature: {stats['llm']['temperature']}\n"
            output += f"- Max Tokens: {stats['llm']['max_tokens']}\n"
            
            return output
        except Exception as e:
            return f"Error fetching stats: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"# {self.title}")
            gr.Markdown(self.description)

            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    value=[],
                    height=500,
                    label="Conversation"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question about your documents...",
                        label="Your Question",
                        scale=4
                    )
                    submit = gr.Button("Send", scale=1)

                with gr.Row():
                    clear = gr.Button("Clear History")

                # Event handlers
                submit.click(
                    fn=self.process_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot]
                )
                msg.submit(
                    fn=self.process_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot]
                )
                clear.click(
                    fn=self.clear_history,
                    outputs=[chatbot]
                )

            with gr.Tab("System Info"):
                stats_output = gr.Markdown()
                refresh_stats = gr.Button("Refresh Stats")
                refresh_stats.click(
                    fn=self.get_stats,
                    outputs=[stats_output]
                )
                # Load stats on tab open
                interface.load(
                    fn=self.get_stats,
                    outputs=[stats_output]
                )

            with gr.Tab("Instructions"):
                gr.Markdown("""
                ## How to Use
                
                1. **Ingest Documents**: First, add your documents to the `data/documents` folder
                2. **Index Documents**: Run the ingestion script to process and index your documents
                3. **Ask Questions**: Use the chat interface to ask questions about your documents
                4. **View Sources**: The system will show which documents were used to generate answers
                
                ## Supported Formats
                - Text files (.txt)
                - PDF files (.pdf)
                - Markdown files (.md)
                - Word documents (.docx)
                
                ## Tips
                - Be specific in your questions
                - The system works best with factual questions about document content
                - If you don't get good answers, try rephrasing your question
                - Check the System Info tab to see how many documents are indexed
                """)

        return interface

    def launch(
        self,
        host: str = "127.0.0.1",
        port: int = 7860,
        share: bool = False
    ):
        """
        Launch the Gradio interface.

        Args:
            host: Host address
            port: Port number
            share: Whether to create public link
        """
        interface = self.create_interface()
        logger.info(f"Launching interface on {host}:{port}")
        interface.launch(
            server_name=host,
            server_port=port,
            share=share
        )
