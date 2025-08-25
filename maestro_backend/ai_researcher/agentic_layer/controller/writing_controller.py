import asyncio
from typing import Dict, Any, Optional

from ai_researcher.agentic_layer.model_dispatcher import ModelDispatcher
from database.models import User

from ai_researcher.core_rag.embedder import TextEmbedder
from ai_researcher.core_rag.reranker import TextReranker

class WritingController:
    """
    Manages user-specific instances of the ModelDispatcher.
    Ensures that each user's requests are handled by a dispatcher
    configured with their specific API keys and settings.
    """

    def __init__(self, user: User, text_embedder: TextEmbedder, text_reranker: TextReranker):
        self.user = user
        self.model_dispatcher = ModelDispatcher(user_settings=user.settings)
        self.text_embedder = text_embedder
        self.text_reranker = text_reranker

    async def run_writing_task(self, prompt: str, context: str) -> str:
        """
        This is a placeholder for the primary method that will orchestrate the writing agent.
        It will use the user-specific ModelDispatcher to interact with the LLM.
        """
        # In a real implementation, this would involve:
        # 1. Creating a SimplifiedWritingAgent.
        # 2. Using the agent to process the prompt and context.
        # 3. The agent would use self.model_dispatcher for all LLM calls.
        
        # For now, this is a simplified placeholder:
        messages = [
            {"role": "system", "content": "You are a helpful writing assistant."},
            {"role": "user", "content": f"Context: {context}\n\nPrompt: {prompt}"}
        ]
        
        response, _ = await self.model_dispatcher.dispatch(messages=messages, agent_mode="writing")
        
        if response and response.choices:
            return response.choices[0].message.content
        
        return "Error: Could not get a response from the model."
