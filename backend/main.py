from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Traffic Vision API")

app.mount(
    "/evidence",
    StaticFiles(directory="evidence"),
    name="evidence"
)

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
    "last_evidence": None,
}

violation_history = []


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
def add_violation(evidence: str | None = None):
    traffic_state["violations"] += 1
    traffic_state["last_evidence"] = evidence

    violation = {
        "id": traffic_state["violations"],
        "traffic_light": traffic_state["traffic_light"],
        "evidence": evidence,
    }

    violation_history.append(violation)

    return {
        "status": "recorded",
        "violations": traffic_state["violations"],
        "evidence": evidence
    }


@app.get("/dashboard")
def dashboard():
    return {
        "traffic_light": traffic_state["traffic_light"],
        "vehicles_detected": traffic_state["vehicles_detected"],
        "violations": traffic_state["violations"],
        "last_evidence": traffic_state["last_evidence"],
        "violation_history": violation_history,
        "system": "online"
    }