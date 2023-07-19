from fastapi import FastAPI
from pydantic import BaseModel
from mycodes import exec_unsup

app = FastAPI()


@app.get('/helloworld')
def get_hello_message():
    return {"message": "Hello World"}


class SchemaOfInputDataPathRequest(BaseModel):
    Path: str
    Num_Clusters: int


class SchemaOfClusteringTableResponse(BaseModel):
    objectData: dict


@app.post('/api/clustering', response_model=SchemaOfClusteringTableResponse)
def getClusteringTable(request_body: SchemaOfInputDataPathRequest):
    # convert to dict
    args = request_body.__dict__
    clusteringTable = exec_unsup(**args)
    return clusteringTable
