import json
from channels.generic.websocket import AsyncWebsocketConsumer
import requests

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        data = json.loads(text_data)
        message = data["message"]

        try:
            response = requests.post(
                "http://fastapi_service:8001/query",
                params={"q": message}
            )

            result = response.json()
            reply = result.get("answer", "No response from AI")

        except Exception as e:
            reply = str(e)

        await self.send(text_data=json.dumps({
            "message": reply
        }))