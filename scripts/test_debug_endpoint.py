import httpx
import asyncio
import traceback

async def test():
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get('http://localhost:8000/api/get-broadcaster-id?channel_name=xqc', timeout=10.0)
            print(f'Status: {r.status_code}')
            print(f'Response: {r.text}')
            if r.status_code == 200:
                data = r.json()
                print(f'SUCCESS! Broadcaster ID: {data.get("broadcaster_id")}')
            else:
                print(f'ERROR: Status {r.status_code}')
    except Exception as e:
        print(f'Exception: {e}')
        traceback.print_exc()

asyncio.run(test())
