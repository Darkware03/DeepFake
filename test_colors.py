import cv2
from app.liveness_engine import validate_light_challenge, decide_liveness
import json

nm = "face.jpg"
ch = "face.jpg"

res = validate_light_challenge(ch, nm, "#FF0033")
print("Features:", list(res.get("features", {}).keys())[:5])

liv = decide_liveness(0.0, res)
breakdown = liv.get("decision_breakdown", {})
raw = breakdown.get("raw_contributions", {})
print("Raw contribs:", raw)

