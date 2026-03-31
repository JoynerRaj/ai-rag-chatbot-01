import asyncio
import websockets
import json

async def test():
    try:
        async with websockets.connect('wss://django-rag.onrender.com/ws/chat/?chat_id=10') as ws:
            print('Connected to PROD')
            payload = json.dumps({"message": "Hello"})
            await ws.send(payload)
            print('Sent payload to PROD')
            
            # Wait for response
            response = await ws.recv()
            print('Received:', response)
            
            # Wait for second response if any? No, only one response
            
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
