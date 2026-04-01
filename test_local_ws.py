import asyncio
import websockets
import json
import socket

async def test():
    try:
        ur = 'ws://127.0.0.1:8080/ws/chat/?chat_id=10'
        async with websockets.connect(ur) as ws:
            print('Connected locally!')
            payload = json.dumps({"message": "Hello from CLI", "document_id": ""})
            await ws.send(payload)
            print('Sent payload')
            
            # Wait for response with timeout
            response = await asyncio.wait_for(ws.recv(), timeout=30.0)
            print('Received:', response)
            
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
