# main.py
import os
import sqlite3
import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ai_engine import generate_response

app = FastAPI()

# 添加 CORS（避免跨域）
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.makedirs("static", exist_ok=True)
DB_PATH = "experiment.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                user_id TEXT PRIMARY KEY,
                history TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )
        """)
        conn.commit()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        yield conn
    finally:
        conn.close()

class LoginRequest(BaseModel):
    username: str

class LoginResponse(BaseModel):
    user_id: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="用户名不能为空")

    with get_db() as conn:
        cur = conn.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            user_id = row[0]
        else:
            user_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO users (user_id, username, created_at) VALUES (?, ?, ?)",
                (user_id, username, now)
            )
            conn.execute(
                "INSERT INTO conversations (user_id, history) VALUES (?, ?)",
                (user_id, "[]")
            )
            conn.commit()
        return LoginResponse(user_id=user_id)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_id = req.user_id
    with get_db() as conn:
        cur = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="用户不存在")

        cur = conn.execute("SELECT history FROM conversations WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        history = json.loads(row[0]) if row else []

        try:
            reply = generate_response(req.message, history)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI error: {str(e)}")

        history.append({"user": req.message, "ai": reply})
        history = history[-30:]

        conn.execute(
            "UPDATE conversations SET history = ? WHERE user_id = ?",
            (json.dumps(history), user_id)
        )
        conn.commit()

        return ChatResponse(reply=reply)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
