from fastapi import FastAPI
from mymodels.my_unsup_cls import main_eval, init_distributed_mode

app = FastAPI()

# initialize distributed mode once
init_distributed_mode(dist_url="env://")

@app.get('/helloworld')
def get_hello_message():
    return {"message": "Hello World"}

    
@app.post('/api/clustering', response_model=SchemaOfTableResponse)
def getClusteringTable():
    params = {
        "pretrained_weights": "/weights/ibot_small_pretrain.pth",
        "data_path": "/dataset"
    }
    
    # convert to dict
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
    
    