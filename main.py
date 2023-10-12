

# from fastapi import FastAPI, Body
# from pydantic import BaseModel
# import base64
# import json
# import numpy as np
# import cv2
# from flora_ai import counting_flowers

# app = FastAPI()



# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


# @app.get("/upload")
# async def upload():
#     return {"message": "Hello World"}
    

# @app.post("/uploadfiles")
# async def upload_files(image_string: str = Body(...)):
    
#     image_data = base64.b64decode(image_string)
#     # Convert the image data to a numpy array
#     nparr = np.frombuffer(image_data, np.uint8)
#     # Decode the numpy array into an image
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     json_str , image =  counting_flowers(image_rgb)


    

   
#     # convert the image to a memory buffer
#     _, buffer = cv2.imencode('.jpg', image)

#     # encode the buffer using base64
#     image_encoded = base64.b64encode(buffer)

#     # convert the encoded buffer to a string
#     image_string = image_encoded.decode('utf-8')

#     json_str = json.dumps(data_contents)
#     # encoded_image = base64.b64encode(image_b64)
#     result =  {
#         "result_image": image_encoded,
#         "result_data": json_str
#     }


   

#     return {"data":result,"message":"message","HTTPstatus":"httpstatus"}


from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
import base64
import cv2
import numpy as np
import json
from flora_ai import counting_flowers

app = FastAPI()



@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/upload")
async def upload():
    return {"message": "Hello World"}





class ImageModel(BaseModel):
    image_string: str = Field(..., example="base64 encoded string")

    @property
    def decode_image(self):
        try:
            return base64.b64decode(self.image_string)
        except Exception:
            raise ValueError("Invalid base64 string")

class ResponseModel(BaseModel):
    data: dict
    message: str
    HTTPstatus: int

@app.post("/uploadfiles", response_model=ResponseModel)
async def upload_files(image: ImageModel):
    try:
        nparr = np.frombuffer(image.decode_image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        json_str, image = counting_flowers(image_rgb)
        _, buffer = cv2.imencode('.jpg', image)
        image_encoded = base64.b64encode(buffer)
        image_string = image_encoded.decode('utf-8')
        result = {
            "result_image": image_encoded,
            "result_data": json_str
        }
        return {"data": result, "message": "Success", "HTTPstatus": 200}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid base64 string")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

