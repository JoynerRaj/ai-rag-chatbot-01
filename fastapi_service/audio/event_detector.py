import os
import time


def _get_mime_type(filename: str) -> str:
    """Return the correct MIME type string based on the audio file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    mime_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "m4a": "audio/mp4",
    }
    return mime_map.get(ext, "audio/mpeg")


def transcribe_audio(filepath: str, filename: str) -> str:
    """
    Upload the audio file to Gemini's Files API and ask it to transcribe
    the spoken content.  Returns the full transcript as a plain string,
    or an empty string if transcription fails or no API key is set.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[transcribe_audio] No GEMINI_API_KEY — skipping transcription.")
        return ""

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    mime_type = _get_mime_type(filename)
    file_obj = None

    try:
        print(f"[transcribe_audio] Uploading {filename!r} ({mime_type}) to Gemini Files API...")
        with open(filepath, "rb") as f:
            file_obj = client.files.upload(
                file=f,
                config=types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name=filename,
                ),
            )

        # Wait until Gemini finishes processing the file (usually a few seconds)
        max_wait = 120  # seconds
        waited = 0
        while hasattr(file_obj, "state") and file_obj.state.name == "PROCESSING":
            if waited >= max_wait:
                print("[transcribe_audio] Timed out waiting for file processing.")
                return ""
            time.sleep(3)
            waited += 3
            file_obj = client.files.get(name=file_obj.name)

        print(f"[transcribe_audio] File ready: {file_obj.uri}")

        # Ask Gemini to transcribe the full audio
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_uri(file_uri=file_obj.uri, mime_type=mime_type),
                types.Part.from_text(
                    "Please provide a complete, verbatim transcription of everything spoken "
                    "in this audio recording. Include all words and sentences. "
                    "Do not summarize — transcribe every spoken word exactly."
                ),
            ],
        )

        transcript = response.text.strip() if response.text else ""
        print(f"[transcribe_audio] Transcription complete: {len(transcript)} chars")
        return transcript

    except Exception as e:
        print(f"[transcribe_audio] Error: {e}")
        return ""
    finally:
        # Always delete the uploaded file from Gemini to keep the account clean
        if file_obj is not None:
            try:
                client.files.delete(name=file_obj.name)
                print(f"[transcribe_audio] Deleted temp file from Gemini: {file_obj.name}")
            except Exception:
                pass


def process_audio(filepath: str) -> list[dict]:
    """
    Processes an audio file and extracts timestamped sound events from it.
    Currently returns mock event data. In production this would use a real
    audio classification model such as YAMNet or CLAP.
    The filepath argument is reserved for future real-model integration.
    """
    mock_events = [
        {"event": "dog bark",    "start_time": "10:05", "end_time": "10:06", "confidence": 0.95},
        {"event": "alarm",       "start_time": "11:00", "end_time": "11:02", "confidence": 0.88},
        {"event": "dog bark",    "start_time": "11:30", "end_time": "11:31", "confidence": 0.91},
        {"event": "door knock",  "start_time": "12:15", "end_time": "12:16", "confidence": 0.76},
        {"event": "glass break", "start_time": "14:20", "end_time": "14:21", "confidence": 0.89},
        {"event": "car horn",    "start_time": "16:45", "end_time": "16:47", "confidence": 0.92},
    ]
    return mock_events
