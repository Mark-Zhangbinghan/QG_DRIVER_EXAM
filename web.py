from fastapi import FastAPI, File, UploadFile
from fastapi import Request
import uvicorn
import json

# 自定函数
my_json = {
    "name": "John",
    "age": 30,
    "city": "New York"
}
json_str = json.dumps(my_json)

print(json_str)
app = FastAPI()


@app.get('/')
async def read_root():
    return {"message": "OK"}


@app.put('/put_car')
async def put_car(the_car_nums: Request):  # 要在url中写参数而不是请求体
    get_car_number = the_car_nums.query_params
    print(get_car_number)
    print(type(get_car_number))
    return {"put succeed"}


@app.get("/get_car_num")
async def get_car_num():  # 要在body中写参数
    return {json_str}


@app.get("/get_position")
async def get_position():  # 要在body中写参数
    return {json_str}


if __name__ == "__main__":
    # uvicorn.run(app="web:app", host="192.168.0.92", port=8080, reload=False)
    uvicorn.run(app="web:app", host="127.0.0.1", port=8080, reload=False)
