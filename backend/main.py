from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Traffic Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

traffic_state = {
    "traffic_light": "RED",
    "vehicles_detected": 0,
    "violations": 0,
}


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "AI Traffic Vision"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/traffic")
def traffic():
    return traffic_state

@app.post("/vehicles/{count}")
def update_vehicles(count: int):
    traffic_state["vehicles_detected"] = count

    return {
        "status": "updated",
        "vehicles_detected": traffic_state["vehicles_detected"]
    }

@app.post("/violation")
def add_violation():
    traffic_state["violations"] += 1

    return {
        "status": "recorded",
        "violations": traffic_state["violations"]
    }

@app.get("/dashboard")
def dashboard():
    return {
        "traffic_light": traffic_state["traffic_light"],
        "vehicles_detected": traffic_state["vehicles_detected"],
        "violations": traffic_state["violations"],
        "system": "online"
    }