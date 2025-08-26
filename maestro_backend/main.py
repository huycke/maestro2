from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

from database.database import SessionLocal, test_connection, init_db
from database import crud
from api import auth, missions, system, chat, chats, documents, websockets, settings, writing, dashboard, admin
from middleware import user_context_middleware

# Configure reduced logging to minimize console noise
from logging_config import setup_logging
setup_logging()  # Will use LOG_LEVEL environment variable
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MAESTRO API",
    description="AI Research Assistant API",
    version="2.0.0-alpha"
)

@app.get("/api/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

# Configure CORS with environment variables
def get_cors_origins():
    """Get CORS allowed origins from environment variables."""
    # Check if we should allow all origins (development mode with nginx proxy)
    allow_wildcard = os.getenv("ALLOW_CORS_WILDCARD", "false").lower() == "true"
    if allow_wildcard:
        logger.info("CORS: Allowing all origins (wildcard mode)")
        return ["*"]
    
    # Build default origins for backward compatibility
    default_origins = [
        "http://localhost",
        "http://localhost:80",
        "http://localhost:3000",
        "http://localhost:3030",
        "http://localhost:5173",
        "http://localhost:8001",
        "http://127.0.0.1",
        "http://127.0.0.1:80",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3030",
        "http://127.0.0.1:8001"
    ]
    
    # Get additional origins from environment variable
    cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if cors_origins_env == "*":
        logger.info("CORS: Allowing all origins via CORS_ALLOWED_ORIGINS=*")
        return ["*"]
    elif cors_origins_env:
        # Split by comma and strip whitespace
        additional_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
        # Combine with defaults, removing duplicates
        all_origins = list(set(default_origins + additional_origins))
        logger.info(f"CORS allowed origins configured: {all_origins}")
        return all_origins
    
    # Also add origins based on old environment variables for backward compatibility
    frontend_host = os.getenv("FRONTEND_HOST")
    backend_host = os.getenv("BACKEND_HOST")
    if frontend_host or backend_host:
        if frontend_host:
            frontend_port = os.getenv("FRONTEND_PORT", "3030")
            default_origins.append(f"http://{frontend_host}:{frontend_port}")
            default_origins.append(f"http://{frontend_host}")
        if backend_host:
            backend_port = os.getenv("BACKEND_PORT", "8001")
            default_origins.append(f"http://{backend_host}:{backend_port}")
            default_origins.append(f"http://{backend_host}")
    
    logger.info(f"CORS allowed origins (defaults): {list(set(default_origins))}")
    return list(set(default_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    # Add max age to reduce preflight requests
    max_age=86400,  # 24 hours
)

# Add user context middleware
app.middleware("http")(user_context_middleware)

# Include API routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(missions.router, prefix="/api", tags=["missions"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(chats.router, prefix="/api", tags=["chats"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(settings.router, prefix="/api", tags=["settings"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(writing.router, tags=["writing"])
app.include_router(websockets.router, tags=["websockets"])
app.include_router(admin.router)

@app.on_event("startup")
async def startup_event():
    """Initialize database, AI components and create first user on startup."""
    # Only log at ERROR level or higher based on LOG_LEVEL setting
    
    # Store the main event loop reference for WebSocket updates from background threads
    from ai_researcher.agentic_layer.context_manager import set_main_event_loop
    set_main_event_loop()
    
    # Initialize database connection and tables
    try:
        # Test database connection
        if not test_connection():
            logger.error("Failed to connect to database")
            raise Exception("Database connection failed")
        
        # Initialize database tables
        init_db()
        logger.info("Database initialized successfully")
        
        # For PostgreSQL, ensure required extensions are available
        if os.getenv("DATABASE_URL", "").startswith("postgresql"):
            from database.init_postgres import ensure_extensions
            ensure_extensions()
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        # Continue anyway, as tables might already exist
    
    # Create a configurable thread pool
    max_workers = int(os.getenv("MAX_WORKER_THREADS", "10"))
    app.state.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    # Initialize singleton components for dependency injection
    try:
        from ai_researcher.agentic_layer.tool_registry import ToolRegistry
        from ai_researcher.core_rag.embedder import TextEmbedder
        from ai_researcher.core_rag.reranker import TextReranker

        app.state.tool_registry = ToolRegistry()
        app.state.text_embedder = TextEmbedder()
        app.state.text_reranker = TextReranker()
        logger.info("Core components (ToolRegistry, TextEmbedder, TextReranker) initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize core components: {e}", exc_info=True)
        # Depending on the severity, you might want to exit the application
        # For now, we'll log the error and continue, but some endpoints might fail.

    # Create first user for development if no users exist
    db = SessionLocal()
    try:
        users = crud.get_users(db)
        if not users:
            from setup_first_user import create_first_user
            create_first_user()
    except Exception as e:
        logger.error(f"Error during initial user check: {e}", exc_info=True)
    finally:
        db.close()
    
    # Clean up dangling CLI-ingested documents
    db = SessionLocal()
    try:
        from database.models import Document, document_group_association
        logger.info("Checking for dangling CLI documents...")
        
        # Find and clean up documents with cli_processing status
        cli_documents = db.query(Document).filter(
            Document.processing_status == "cli_processing"
        ).all()
        
        if cli_documents:
            logger.info(f"Found {len(cli_documents)} dangling CLI documents, cleaning up...")
            deleted_count = 0
            
            for doc in cli_documents:
                try:
                    # Delete associated files
                    if doc.file_path and os.path.exists(doc.file_path):
                        os.remove(doc.file_path)
                    
                    markdown_path = f"/app/data/markdown_files/{doc.id}.md"
                    if os.path.exists(markdown_path):
                        os.remove(markdown_path)
                    
                    # Remove from document groups
                    db.execute(
                        document_group_association.delete().where(
                            document_group_association.c.document_id == doc.id
                        )
                    )
                    
                    # Delete document record
                    db.delete(doc)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to clean up document {doc.id}: {e}")
            
            db.commit()
            logger.info(f"Cleaned up {deleted_count} dangling CLI documents")
        else:
            logger.debug("No dangling CLI documents found")
            
    except Exception as e:
        logger.error(f"Failed to run CLI document cleanup: {e}", exc_info=True)
        # Don't fail startup if cleanup fails
    finally:
        db.close()
    
    # Initialize AI research components
    try:
        from api.missions import initialize_ai_components
        success = initialize_ai_components(app)
        if not success:
            logger.error("Failed to initialize AI research components")
    except Exception as e:
        logger.error(f"Error during AI component initialization: {e}", exc_info=True)
    
    # DISABLED: Consistency check causes crashes at startup
    # The consistency manager has complex dependencies that fail during initialization
    # TODO: Fix the initialization order before re-enabling
    # try:
    #     from services.document_consistency_monitor import consistency_monitor
    #     asyncio.create_task(consistency_monitor.force_consistency_check())
    #     logger.info("Running one-time document consistency check at startup")
    # except ImportError as e:
    #     logger.warning(f"Could not import consistency monitor: {e}")
    # except Exception as e:
    #     logger.error(f"Failed to run consistency check: {e}", exc_info=True)

    # Start the background document processor
    try:
        from services.background_document_processor import background_processor
        import threading

        # Run the processor in a daemon thread
        processor_thread = threading.Thread(target=background_processor.start, daemon=True)
        processor_thread.start()

        app.state.background_processor = background_processor
        app.state.processor_thread = processor_thread
        logger.info("Background document processor started.")

    except Exception as e:
        logger.error(f"Failed to start background document processor: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Only log at ERROR level or higher based on LOG_LEVEL setting
    if hasattr(app.state, "thread_pool"):
        app.state.thread_pool.shutdown(wait=True)
    
    # Gracefully shut down the background processor
    if hasattr(app.state, "background_processor"):
        logger.info("Shutting down background document processor...")
        app.state.background_processor.shutdown()
        # It's a daemon thread, so we don't strictly need to join it

    # No need to stop monitoring since we only run once at startup
    pass

@app.get("/")
def read_root():
    return {
        "message": "MAESTRO API v2.0",
        "status": "running",
        "docs": "/docs"
    }

