import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as c:
        r = await c.get('http://localhost:8000/api/get-broadcaster-id?channel_name=xqc')
        print(f'Status: {r.status_code}')
        print(f'Response: {r.text}')
        if r.status_code == 200:
            data = r.json()
            print(f'Broadcaster ID: {data.get("broadcaster_id")}')

asyncio.run(test())

