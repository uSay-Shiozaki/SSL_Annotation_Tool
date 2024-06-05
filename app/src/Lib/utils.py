import asyncio
import requests
import os
import threading
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from tkinter import filedialog
import tkinter as tk

def getClusteringTable(obj):
    endPoint: str = 'http://kivy_server:8000/api/clustering'
    resp = requests.post(endPoint, json=obj)
    
    if resp.status_code == requests.codes.ok:
        resp = resp.json()
        return resp['body']
    else:
        return "err"

def runSSL(obj):
    res = getClusteringTable(obj) 
    
    return res

def saveDialog():

    typ = [('JSON File', '*.json'), ('All', '*')]
    root = tk.Tk()
    root.withdraw()
    file = filedialog.asksaveasfilename(filetypes=typ)
    print(file)
    return file

def openDialog():

    typ = [('JSON File', '*.json'), ('All', '*')]
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(filetypes=typ)
    print(file)
    return file

def openDirDialog():

    root = tk.Tk()
    root.withdraw()
    file = filedialog.askdirectory()
    print(file)
    return file
