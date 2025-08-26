from fastapi import Request, Depends
from ai_researcher.agentic_layer.tool_registry import ToolRegistry
from ai_researcher.core_rag.embedder import TextEmbedder
from ai_researcher.core_rag.reranker import TextReranker
from ai_researcher.agentic_layer.controller.writing_controller import WritingController
from auth.dependencies import get_current_user_from_cookie
from database.models import User

def get_tool_registry(request: Request) -> ToolRegistry:
    """Get the singleton ToolRegistry instance from the app state."""
    return request.app.state.tool_registry

def get_text_embedder(request: Request) -> TextEmbedder:
    """Get the singleton TextEmbedder instance from the app state."""
    return request.app.state.text_embedder

def get_text_reranker(request: Request) -> TextReranker:
    """Get the singleton TextReranker instance from the app state."""
    return request.app.state.text_reranker

def get_writing_controller(
    request: Request,
    current_user: User = Depends(get_current_user_from_cookie),
    text_embedder: TextEmbedder = Depends(get_text_embedder),
    text_reranker: TextReranker = Depends(get_text_reranker),
) -> WritingController:
    """Get a user-specific WritingController instance."""
    return WritingController(
        user=current_user,
        text_embedder=text_embedder,
        text_reranker=text_reranker,
    )
