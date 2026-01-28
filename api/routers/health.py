"""
Health check and system status endpoints
"""
from fastapi import APIRouter
from config import get_settings
import time

router = APIRouter()
settings = get_settings()


@router.get("/health")
async def health_check():
    """Basic health check - database disabled, using arXiv API"""
    redis_status = "not tested"
    
    try:
        from utils.redis_client import get_redis_client
        redis = get_redis_client()
        redis.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unavailable: {str(e)}"
    
    return {
        "status": "operational",
        "timestamp": int(time.time()),
        "services": {
            "api": "healthy",
            "arxiv_search": "enabled",
            "redis": redis_status,
            "llm_provider": settings.llm_provider
        }
    }


@router.get("/status")
async def system_status():
    """Detailed system status - copilot mode (no database)"""
    
    return {
        "environment": settings.environment,
        "mode": "research_copilot",
        "note": "Database features disabled - using live arXiv API",
        "configuration": {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "storage_type": settings.storage_type
        }
    }
