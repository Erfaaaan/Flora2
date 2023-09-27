

from fastapi import FastAPI, Body
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


@app.get("/upload")
async def upload():
    return {"message": "Hello World"}
    

@app.post("/uploadfiles")
async def upload_files(image_string: str = Body(...)):
    # return "done"
    # Decode the Base64 string
    image_data = base64.b64decode(image_string)
    # Convert the image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    # Decode the numpy array into an image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    data_contents = {"test":"test"}

    #image_b64 = base64.b64encode(image_contents).decode("utf-8")

    # convert the image to a memory buffer
    _, buffer = cv2.imencode('.jpg', image)

    # encode the buffer using base64
    image_encoded = base64.b64encode(buffer)

    # convert the encoded buffer to a string
    image_string = image_encoded.decode('utf-8')

    json_str = json.dumps(data_contents)
    # encoded_image = base64.b64encode(image_b64)
    result =  {
        "result_image": image_encoded,
        "result_data": json_str
    }


    #result = AIModel.process_image(image_encoded, data_contents)

    return {"data":result,"message":"message","HTTPstatus":"httpstatus"}

