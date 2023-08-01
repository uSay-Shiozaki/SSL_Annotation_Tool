import asyncio
import requests
import os
import threading
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.button import Button

def getClusteringTable():
    endPoint: str = 'http://ssl_server:8000/api/clustering'
    
    resp = requests.post(endPoint)
    
    if resp.status_code == requests.codes.ok:
        resp = resp.json()
        return resp['body']
    else:
        return "err"

def runSSL():
    res = getClusteringTable() 
    
    return res

def openInfoPopup(_button: Button):
    def update_progress(an_object, popup):
        for i in range(10, -1, -1):
            time.sleep(1.0)
            print("progress: {}".format(i))
            an_object.value = i
        popup.dismiss()
    
    content = ProgressBar(max=10)
    popup = Popup(
        title='Progress',
        size_hint=(None, None), size=(400, 180),
        content=content,
        auto_dismiss=False
    )
    threading.Thread(
        target=partial(
            update_progress, 
            an_object=content,
            popup=popup,
        ),
        daemon=True
        ).start()
    popup.open()
            