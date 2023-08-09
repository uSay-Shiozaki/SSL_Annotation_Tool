from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mymodels.ibot.my_unsup_cls import main_eval, init_distributed_mode
from http_types import (
    SchemaOfSmSLwithiBOTRequest,
    SchemaOfSmSLwithSwAVRequest,
    SchemaOfTableResponse,
    SchemaOfInputDataPathRequest
)
app = FastAPI()

# initialize distributed mode once
init_distributed_mode(dist_url="env://")

@app.get('/helloworld')
def get_hello_message():
    return {"message": "Hello World"}

@app.exception_handler(RequestValidationError)
async def handler(request: Request, exc:RequestValidationError):
    print(exc)
    return JSONResponse(content={}, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

    
@app.post('/api/clustering', response_model=SchemaOfTableResponse)
def getClusteringTable(params: SchemaOfInputDataPathRequest):
    # parse dict args
    params = params.dict()
    clusteringTable = main_eval(**params)
    return clusteringTable


@app.post('/api/smsl-ibot', response_model=SchemaOfTableResponse)
def getSmSLwithiBOT():
    params = {
        "pretrained_weights": "/weights/ibot_small_pretrain.pth",
        "data_path": "/dataset"
    }
    params = SchemaOfSmSLwithiBOTRequest(**params)
    
    smslTable = main_eval(params)
    return smslTable

@app.post('/api/smsl-swav', response_model=SchemaOfTableResponse)
def getSmSLwithSwAV():
    params = {
        "pretrained_weights": "/weights/swav_pretrain.pth",
        "data_path": "/dataset"
    }
    params = SchemaOfSmSLwithSwAVRequest(**params)
    smslTable = ""
    return smslTable
    
    