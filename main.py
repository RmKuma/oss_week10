from fastapi import File, UploadFile, Request, FastAPI
from fastapi.templating import Jinja2Templates
import base64
from test import *
from io import BytesIO

PATH = './cifar_net.pth'
net = ImageClassifier()
net.load_state_dict(torch.load(PATH, map_location ='cpu'))
net.eval()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
  
@app.post("/upload")
def upload(request: Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    base64_encoded_image = base64.b64encode(contents).decode("utf-8")
    img = Image.open(BytesIO(contents))
    _input = torch.stack([transform(img)])
    output = net(_input).argmax().tolist()

    if output == 0:
        result = "Cat"
    else:
        result = "Dog"

    return templates.TemplateResponse("display.html", {"request": request, "result":result, "myImage":base64_encoded_image})
