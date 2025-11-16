import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as c:
        r = await c.get('http://localhost:8000/api/get-broadcaster-id?channel_name=xqc')
        print(f'Status: {r.status_code}')
        print(f'Full Response: {r.text}')
        print(f'Headers: {dict(r.headers)}')
        if r.status_code == 500:
            try:
                import json
                data = r.json()
                print(f'Error Detail: {data.get("detail")}')
            except:
                pass

asyncio.run(test())
