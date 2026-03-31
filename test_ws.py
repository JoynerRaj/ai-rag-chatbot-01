import asyncio
import websockets

async def test():
    try:
        async with websockets.connect('ws://127.0.0.1:8089/ws/chat/?chat_id=10') as ws:
            print('Connected')
            await ws.send('{"type": "ping"}')
            print('Sent ping')
    except Exception as e:
        print('Error:', e)

asyncio.run(test())
