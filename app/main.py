from fastapi import FastAPI
from app.core import logger
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CreditFlow API starting...")
    yield
    logger.info("CreditFlow API shutting down...")


app = FastAPI(
    title="CreditFlow AI",
    lifespan=lifespan,
    version="0.1.0",
    description="Production-grade Credit Risk Decision Platform",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CreditFlow API starting...")
    yield
    logger.info("CreditFlow API shutting down...")


app = FastAPI(
    title="CreditFlow AI",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}
