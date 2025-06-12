"""
Encite AI-Powered Scheduling Platform
Main FastAPI Application Entry Point
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

#An ASGI server used to run FastAPI applications. It’s not for caching or database, but for serving your app
import uvicorn
#lasses for hadnling HTTP requests and responses, including CORS and GZip middleware
from fastapi import FastAPI, Request, HTTPException
#Middleware that allows api to be called cross browser, cross-origin resource sharing
from fastapi.middleware.cors import CORSMiddleware
#Middleware for compressing responses to reduce bandwidth usage
from fastapi.middleware.gzip import GZipMiddleware
#Utility to return JSON-formatted HTTPs responses
from fastapi.responses import JSONResponse
#kd

import torch

# Internal imports
from config.settings import get_settings
from config.database import init_databases
from app.api.routes import auth, users, groups, scheduling, recommendations, feedback
from app.api.middleware.auth import AuthMiddleware
from app.api.middleware.rate_limiting import RateLimitMiddleware
from app.api.middleware.logging import LoggingMiddleware
from app.services.ai_inference.graph_inference import GraphInferenceService
from app.services.ai_inference.retrieval_service import RetrievalService
from app.services.ai_inference.rl_optimization import RLOptimizationService
from app.services.vector_service import VectorService
from app.services.cache_service import CacheService
from app.utils.auth_utils import setup_firebase_auth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
services: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting Encite AI Scheduling Platform...")
    
    try:
        # Load configuration
        settings = get_settings()
        logger.info(f"Environment: {settings.ENVIRONMENT}")
        
        # Initialize databases
        await init_databases()
        logger.info("✅ Databases initialized")
        
        # Initialize Firebase Auth
        setup_firebase_auth()
        logger.info("✅ Firebase Auth configured")
        
        # Initialize vector service (Pinecone)
        services['vector'] = VectorService()
        await services['vector'].initialize()
        logger.info("✅ Vector service (Pinecone) initialized")
        
        # Initialize cache service (Redis)
        services['cache'] = CacheService()
        await services['cache'].initialize()
        logger.info("✅ Cache service (Redis) initialized")
        
        # Load AI models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize HGT Graph Model
        services['graph_model'] = GraphInferenceService(device=device)
        await services['graph_model'].load_model()
        logger.info("✅ HGT Graph model loaded")
        
        # Initialize Two-Tower Retrieval Model
        services['retrieval_model'] = RetrievalService(device=device)
        await services['retrieval_model'].load_model()
        logger.info("✅ Two-Tower retrieval model loaded")
        
        # Initialize PPO RL Optimizer
        services['rl_optimizer'] = RLOptimizationService()
        await services['rl_optimizer'].load_model()
        logger.info("✅ PPO RL optimizer loaded")
        
        # Store services in app state
        app.state.services = services
        
        logger.info("🎉 All services initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("🛑 Shutting down Encite platform...")
    
    # Cleanup services
    for service_name, service in services.items():
        try:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
            logger.info(f"✅ {service_name} service cleaned up")
        except Exception as e:
            logger.error(f"❌ Error cleaning up {service_name}: {e}")
    
    logger.info("👋 Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="Encite AI Scheduling Platform",
    description="AI-powered social scheduling with deep learning, graph intelligence, and reinforcement learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthMiddleware)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": request.url.path
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring
    """
    try:
        # Check service availability
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": {}
        }
        
        # Check individual services
        for service_name, service in app.state.services.items():
            try:
                if hasattr(service, 'health_check'):
                    is_healthy = await service.health_check()
                    health_status["services"][service_name] = "healthy" if is_healthy else "unhealthy"
                else:
                    health_status["services"][service_name] = "healthy"
            except Exception as e:
                health_status["services"][service_name] = f"unhealthy: {str(e)}"
        
        # Overall health
        unhealthy_services = [name for name, status in health_status["services"].items() 
                            if not status.startswith("healthy")]
        
        if unhealthy_services:
            health_status["status"] = "degraded"
            health_status["unhealthy_services"] = unhealthy_services
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
        )

# API Routes
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(groups.router, prefix="/api/v1/groups", tags=["Groups"])
app.include_router(scheduling.router, prefix="/api/v1/schedule", tags=["Scheduling"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["Feedback"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Encite AI Scheduling Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        workers=1 if settings.ENVIRONMENT == "development" else 4,
        log_level="info"
    )