from fastapi import FastAPI
from pydantic import BaseModel
from mymodels.my_unsup_cls import main_eval, init_distributed_mode

app = FastAPI()

# initialize distributed mode once
init_distributed_mode(dist_url="env://")

@app.get('/helloworld')
def get_hello_message():
    return {"message": "Hello World"}

class SchemaOfInputDataPathRequest(BaseModel):
    pretrained_weights: str
    data_path: str

class SchemaOfClusteringTableResponse(BaseModel):
    body: dict

@app.post('/api/clustering', response_model=SchemaOfClusteringTableResponse)
def getClusteringTable():
    args = {
        "pretrained_weights": "/weights/ibot_small_pretrain.pth",
        "data_path": "/dataset"
    }
    
    # convert to dict
    clusteringTable = main_eval(**args)
    return clusteringTable
