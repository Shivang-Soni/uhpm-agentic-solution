from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="UHPM Agent API")

app.include_router(router)


@app.get("/")
def root():
    return {"status": "ok", "msg": "UHPM Agent API is running."}
