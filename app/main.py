import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import vision
from . import state

app = FastAPI(title="Visual POS System")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Static files
static_dir = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Templates
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)


# -------------------------
# APPLICATION EVENTS
# -------------------------

@app.on_event("startup")
async def startup_event():
    print("[APP] Starting camera...")
    vision.start_camera()


@app.on_event("shutdown")
async def shutdown_event():
    print("[APP] Stopping camera...")
    vision.stop_camera()


# -------------------------
# ROUTES
# -------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    context = {
        "request": request,
        "session_status": "WAITING FOR SESSION",
        "items": [],
        "total_amount": 0.0,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/video_feed")
async def video_feed():

    def gen():
        while True:
            frame_bytes = vision.get_latest_frame_jpeg()
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            else:
                time.sleep(0.05)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/pos_state")
async def pos_state():
    return JSONResponse(content=state.get_pos_state_dict())
