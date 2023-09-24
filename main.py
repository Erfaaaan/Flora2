

from fastapi import FastAPI, UploadFile, File, Form , Response
from pydantic import BaseModel
import base64
import json
import numpy as np
import cv2


app = FastAPI()

class AIModel:
    @staticmethod
    def process_image(image_b64: str,data_json: dict) -> dict:
        # This is where your AI detection logic goes.
        # For this example, we'll just return the input as output.
        json_str = json.dumps(data_json)
        #encoded_image = base64.b64encode(image_b64)
        return {
            "result_image": image_b64,
            "result_data": json_str
        }

@app.get("/")
async def root():
    return {"message": "Hello World"}


