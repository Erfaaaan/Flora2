

from fastapi import FastAPI, UploadFile, File, Form , Response
from pydantic import BaseModel
import base64
import json
import numpy as np
import cv2


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


