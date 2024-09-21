import asyncio
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import numpy as np

# 自定义引用
from CAV.code.position_give import get_data
from add_json import sub_switch_road
import share

micro_app = FastAPI()


@micro_app.put("/put_sub_position")
async def post_sub_position(path_request: Request):
    path_json = await path_request.json()
    share.array_list = get_data(path_json)
    return {"put succeed"}


@micro_app.get("/get_sub_position")
async def get_sub_position():
    final_json_list = sub_switch_road(share.array_list)
    final_length = len(final_json_list)
    print("sub_car_t:")
    print("cnt/len")
    print(share.switch_cnt + 1, "/", final_length)
    the_switch_json = final_json_list[share.switch_cnt]
    share.switch_cnt += 1
    return the_switch_json


@micro_app.get("/reset_counter")
async def reset_counter():
    share.switch_cnt = 0
    return {"message": "Counter has been reset."}
