from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

appl = FastAPI()

fe_origin = ['http://localhost','http://localhost:3000']

appl.add_middleware(CORSMiddleware,
                    allow_origins = fe_origin,
                    allow_credentials = True,
                    allow_methods = ['*'],
                    allow_headers = ['*'])

model = tf.keras.models.load_model('../models/apples')
classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

@appl.get("/ping")
async def ping():
    return 'Testing the API'

def convert_to_array(data)->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@appl.post("/predict")
async def predict(file:UploadFile = File(...)):
    image = convert_to_array(await file.read())
    batch = np.expand_dims(image,0)
    prediction = model.predict(batch)
    prediction_class = classes[np.argmax(prediction[0])]
    confidence = str(np.max(prediction[0])*100) + '%'
    return {
        'predicted_class' : prediction_class,
        'confidence' : confidence
    }

if __name__ == '__main__':
    uvicorn.run(appl, host='localhost', port=1000)