from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/")
async def hello_world():
    return {"message": "Hello World"}


@app.get("/hello")
async def hello():
    return {"message": "Hello"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

