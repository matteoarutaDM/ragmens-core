"""Adapter FastAPI opzionale; passare una pipeline già preparata."""
def create_app(pipeline):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    @asynccontextmanager
    async def lifespan(app):
        yield
        pipeline.close()
    app=FastAPI(title='ragmens-core',lifespan=lifespan)
    class Request(BaseModel):
        question: str
        stream: bool = False
    # FastAPI deve poter risolvere l'annotazione locale anche con future annotations.
    @app.get('/health')
    def health(): return {'status':'ok'}
    @app.post('/chat')
    def chat(request: Request):
        if not request.question.strip(): raise HTTPException(status_code=400,detail='Domanda vuota')
        try:
            if request.stream: return StreamingResponse(pipeline.ask_stream(request.question),media_type='text/plain')
            return pipeline.ask(request.question)
        except ValueError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
        except Exception as exc: raise HTTPException(status_code=503,detail='Servizio RAG non disponibile') from exc
    return app
