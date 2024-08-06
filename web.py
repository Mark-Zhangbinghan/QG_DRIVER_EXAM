from fastapi import FastAPI, File, UploadFile
from fastapi import Request
import uvicorn
import json
#自定函数
from add_json import get_car_data
my_json = {
    "name": "John",
    "age": 30,
    "city": "New York"
}
json_str = json.dumps(my_json)

print(json_str)
app = FastAPI()


@app.post('/')
async def read_root():
    return {"message": "OK"}


@app.get("/get_position")
async def get_position():  # 要在body中写参数
    return {json_str}


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="192.168.0.92", port=8080, reload=False)
