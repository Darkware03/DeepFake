import asyncio
import httpx
import os

async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Checking /health")
        resp = await client.get("http://localhost:8086/health")
        print("Health Response:", resp.json())

        print("Testing POST /api/v1/detect (Image only)")
        with open("dummy_image.png", "rb") as f:
            resp = await client.post("http://localhost:8086/api/v1/detect", files={"file": ("dummy_image.png", f, "image/png")})
            print("Detect Image:", resp.json())

        print("Testing POST /api/v1/detect (Video only)")
        with open("dummy_video.mp4", "rb") as f:
            resp = await client.post("http://localhost:8086/api/v1/detect", files={"video": ("dummy_video.mp4", f, "video/mp4")})
            print("Detect Video:", resp.json())
            
        print("Testing POST /api/v1/detect (Video with challenge)")
        with open("dummy_video.mp4", "rb") as f, open("dummy_image.png", "rb") as ch, open("dummy_image.png", "rb") as nm:
            resp = await client.post("http://localhost:8086/api/v1/detect", 
                files={
                    "video": ("dummy_video.mp4", f, "video/mp4"),
                    "challenge_image": ("challenge.png", ch, "image/png"),
                    "normal_image": ("normal.png", nm, "image/png")
                },
                data={
                    "challenge_color": "#FF0000"
                }
            )
            print("Detect Challenge:", resp.json())

if __name__ == "__main__":
    asyncio.run(main())