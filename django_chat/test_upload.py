import os
from google import genai
from google.genai import types
os.environ['GEMINI_API_KEY'] = 'AIzaSyAv5MF55CZiTCLHGVJk31tvoQZB4Ayu2-U'
client = genai.Client()
with open('test.mp3', 'wb') as f:
    f.write(b'test')
try:
    with open('test.mp3', 'rb') as f:
        file_obj = client.files.upload(file=f, config=types.UploadFileConfig(mime_type="audio/mpeg", display_name="test.mp3"))
    print("OK", getattr(file_obj, 'name', 'No name'))
except Exception as e:
    import traceback
    traceback.print_exc()
