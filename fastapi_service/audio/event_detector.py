import uuid

def process_audio(file_bytes: bytes) -> list[dict]:
    """
    Simulates processing an audio file and extracting events.
    In a real production environment, this would utilize a model like YAMNet or CLAP.
    """
    # We return mock data for demonstration purposes
    # In a real app, we'd chunk `file_bytes` and pass through a classifier
    mock_events = [
        {"event": "dog bark", "start_time": "10:05", "end_time": "10:06", "confidence": 0.95},
        {"event": "alarm", "start_time": "11:00", "end_time": "11:02", "confidence": 0.88},
        {"event": "dog bark", "start_time": "11:30", "end_time": "11:31", "confidence": 0.91},
        {"event": "door knock", "start_time": "12:15", "end_time": "12:16", "confidence": 0.76},
        {"event": "glass break", "start_time": "14:20", "end_time": "14:21", "confidence": 0.89},
        {"event": "car horn", "start_time": "16:45", "end_time": "16:47", "confidence": 0.92}
    ]
    return mock_events
