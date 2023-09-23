# from fastapi import FastAPI,File, UploadFile
#
# app = FastAPI()
#
#
#
# @app.post("/upload-image")
# async def upload_image(image: UploadFile = File(...)):
#     # Process the image and generate the output
#     processed_image, json_data = process_image(image)
#
#     return {"processed_image": processed_image, "json_data": json_data}
#
#
# def process_image(image: UploadFile):
#     # Load the image
#     img = Image.open(image.file)
#
#     # Run your AI model on the image and get the processed image and JSON data
#     processed_image = your_ai_model(img)
#     json_data = generate_json(processed_image)
#
#     return processed_image, json_data
#
# def generate_json(processed_image):
#     # Generate the JSON data based on the processed image
#     json_data = {
#         "metadata": {
#             "width": processed_image.width,
#             "height": processed_image.height
#         },
#         # Add any other relevant data from your AI model
#     }
#
#     return json.dumps(json_data)
#
#
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Form
# import base64
#
# app = FastAPI()
#
# @app.post("/process_image")
# def process_image(image: str = Form(...)):
#     # Decode the base64-encoded image
#     image_bytes = base64.b64decode(image)
#
#     # Process the image using your AI model
#     #processed_image = process_image_with_ai(image_bytes)
#     processed_image = 0
#     # Generate the JSON file using your AI model
#     #json_data = generate_json_with_ai(processed_image)
#     json_data = 0
#     # Encode the processed image and JSON file as base64 strings
#     encoded_image = base64.b64encode(processed_image).decode("utf-8")
#     #encoded_json = base64.b64encode(json_data.encode("utf-8")).decode("utf-8")
#
#     # Return the base64-encoded processed image and JSON file
#     return {"image": encoded_image, "json": json_data}
#
#
#


from fastapi import FastAPI, UploadFile, File, Form
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

@app.post("/uploadfiles/")
async def upload_files(image_string: str = Form(...)):
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