"""
audio_service.py

Handles everything related to audio file transcription via the Gemini Files API.
Keeping this separate from views.py means the transcription logic can be tested
and reused without pulling in any Django HTTP machinery.
"""

import os
import time

from google import genai
from google.genai import types


# Maps audio file extensions to the MIME type Gemini expects
AUDIO_MIME_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
}


def transcribe_audio(filepath: str, filename: str) -> str:
    """
    Upload an audio file to the Gemini Files API and return a verbatim transcript.

    The function waits up to 120 seconds for Gemini to finish processing the file,
    then asks the model for a word-for-word transcription of everything spoken.

    Returns an empty string if no speech was detected or if anything goes wrong.
    Raises an exception if the API call itself crashes so the caller can log it.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set — check your environment variables")

    ext = filename.lower().rsplit(".", 1)[-1]
    mime_type = AUDIO_MIME_TYPES.get(ext, "audio/mpeg")

    client = genai.Client(api_key=api_key)
    uploaded_file = None

    try:
        print(f"[audio] uploading {filename!r} to Gemini Files API...")
        with open(filepath, "rb") as f:
            uploaded_file = client.files.upload(
                file=f,
                config=types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name=filename,
                ),
            )

        # Gemini processes the file asynchronously — poll until it's ready
        waited_seconds = 0
        while hasattr(uploaded_file, "state") and uploaded_file.state.name == "PROCESSING":
            if waited_seconds >= 120:
                print("[audio] gave up waiting after 120 seconds")
                return ""
            time.sleep(3)
            waited_seconds += 3
            uploaded_file = client.files.get(name=uploaded_file.name)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=mime_type),
                types.Part.from_text(
                    text=(
                        "Provide a complete, verbatim transcription of all speech in this audio. "
                        "Include every word — do not summarize or skip anything."
                    )
                ),
            ],
        )

        transcript = response.text.strip() if response.text else ""
        print(f"[audio] transcription done — {len(transcript)} characters")
        return transcript

    except Exception as e:
        import traceback
        print(f"[audio] transcription failed for {filename!r}: {e}\n{traceback.format_exc()}")
        raise

    finally:
        # Remove the file from Gemini's servers as soon as we're done with it
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
