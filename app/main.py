from fastapi import FastAPI
from app.core import logger

app = FastAPI(
    title="CreditFlow AI",
    version="0.1.0",
    description="Production-grade Credit Risk Decision Platform",
)


# @app.get("/")
# def root():
#    return {"message": "CreditFlow AI is running 🚀"}


# @app.get("/health")
# def health():
#    return {
#        "status": "healthy",
#        "service": "creditflow-ai",
#        "version": "0.1.0",
#    }


@app.on_event("startup")
async def startup():
    logger.info("CreditFlow API started")


@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}
