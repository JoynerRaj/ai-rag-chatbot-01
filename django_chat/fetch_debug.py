import urllib.request
import json
try:
    url = "https://django-rag.onrender.com/debug/embed/"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(json.dumps(data.get("last_failed_document", {}), indent=2))
except Exception as e:
    print(e)
