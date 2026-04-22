import os
import time


MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
}

MOCK_EVENTS = [
    {"event": "dog bark",    "start_time": "10:05", "end_time": "10:06", "confidence": 0.95},
    {"event": "alarm",       "start_time": "11:00", "end_time": "11:02", "confidence": 0.88},
    {"event": "dog bark",    "start_time": "11:30", "end_time": "11:31", "confidence": 0.91},
    {"event": "door knock",  "start_time": "12:15", "end_time": "12:16", "confidence": 0.76},
    {"event": "glass break", "start_time": "14:20", "end_time": "14:21", "confidence": 0.89},
    {"event": "car horn",    "start_time": "16:45", "end_time": "16:47", "confidence": 0.92},
]


def process_audio(filepath):
    """
    Detect sound events in an audio file.
    Currently returns mock data; swap in YAMNet or CLAP here when ready.
    """
    return MOCK_EVENTS
