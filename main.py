from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import os
import uvicorn
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

from solver import solve_quiz, set_log_callback

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

MY_SECRET_KEY = os.getenv("MY_SECRET_KEY", "my_secret_string")

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Log callback to bridge solver -> websocket
def log_to_ws(message: str, level: str = "info", extra: dict = None):
    payload = {"type": "log", "message": message, "level": level}
    if extra:
        payload.update(extra)
    
    # We need to run this async broadcast from sync/async context
    # Since solver is async, we can just await if we pass an async callback,
    # or use a global loop. For simplicity, we'll try to get the running loop.
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            loop.create_task(manager.broadcast(payload))
    except RuntimeError:
        # If no loop (shouldn't happen in FastAPI), ignore
        pass

# Register callback
set_log_callback(log_to_ws)

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

class QuizRequest(BaseModel):
    email: str
    secret: str
    url: HttpUrl

@app.post("/quiz")
async def quiz_endpoint(request: QuizRequest, background_tasks: BackgroundTasks):
    if request.secret != MY_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    # Trigger background task
    background_tasks.add_task(solve_quiz, str(request.url), request.email, request.secret)
    
    return {"message": "Quiz started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
