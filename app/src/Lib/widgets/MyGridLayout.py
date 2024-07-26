import os
import sys
from kivymd.uix.gridlayout import MDGridLayout
from kivy.uix.label import Label
import json
from widgets.MySmartTile import MySmartTile
import random
import math
import logging
from widgets.Timer import Timer
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooser, FileChooserListLayout, FileChooserIconLayout
from tkinter import filedialog
import asyncio
from dotenv import load_dotenv
from kivy.network.urlrequest import UrlRequest
import threading
import requests
import utils
from widgets import PopupRaiseError

load_dotenv()
DIALOG_DEFAULT_PATH = "/database"
SAVEDIR = '/database'
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

class MyGridLayout(MDGridLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    
    def __init__(self, quantity=240, **kwargs):
        super().__init__(**kwargs)
        # self.list = self.getFileList(IMAGE_DIR,quantity)
        # update self.root in KVfile
        self.root = None
        self.quantity = quantity
        self.startId = 0
        self.pageId = 0
        self.tileId = 0
        self.tiles = []
        self.pressButtonList = []
        self.selectFilePathList = []
        self.tilesRemain = []
        self.selectSave = False
        self.clustering = True
        self.modeRemain = False
        self.modeText = "Remove\n Target"
        self.index = 0
        self.nodeNmb = 0
        self.nodeList = []
        self.semiBool = False
        self.fileList = []
        self.jsons = None
        logging.info("GRID LAUNCHED")

        # print(f"self.fileList is below\n {self.fileList}")

        self.timer = Timer()
        self.previewSrc = ""
        Clock.schedule_interval(self.update, 0.1)
        
    def update(self, dt):

        # self.root.ids.preview.reload()
        pass
        
    def change_save_mode(self):
        self.selectSave = not self.selectSave
        # reset list of pushed tiles
        for tile in self.pressButtonList:
            tile.canvas.after.remove(tile.color)
            tile.canvas.after.remove(tile.rect)
            tile.press = True
            
        if not self.selectSave:
            self.modeText = "Remove\n Target"
            self.root.ids.mode_change.background_color = "black"
        else:
            self.modeText = "Save\n Target"
            self.root.ids.mode_change.background_color = "red"
        print(f"mode is {self.modeText} now")
        self.root.ids.mode_change.text = self.modeText

    def start(self):
        self.clear_all()
        self.semiBool = False
        self.selectSave = False
        self.clustering = True
        self.tilesRemain = [] 

        success = self.openFile()
        if success:
            self.show_node(
                self.startId, self.quantity, self.nodeNmb, clustering=self.clustering
            )

    def semi_learning_button(self):
        self.clustering = False
        self.selectSave = False
        self.targetList = self.random_extract(10, self.fileList)
        logging.info(self.targetList)
        self.clear_all()
        self.show_list(self.startId, self.quantity, self.targetList)
        self.semiBool = True

    def start_semi_learning(self):
        self.semiBool = False

    def show_remain(self):
        self.modeRemain = True
        logging.info("Calling show_remain()")
        self.clear_all()
        logging.info(self.tilesRemain)
        self.startId = 0
        self.show_list(self.startId, self.quantity, self.tilesRemain)

    def semi_next_button(self):
        '''
        this button for move to next page of the list of random extracted
        images for self-supervised learning. This contains chosen images in 
        "self.targetList".
        '''

        self.targetList = []
        logging.info("Calling semi-next button")
        if self.startId > self.len:
            self.startId = self.len
            return

        if self.targetList is None:
            self.add_widget(Label(text="No Images"))
        else:
            self.clear_all()
            self.startId += self.quantity
            # print(self.startId)

            self.show_list(self.startId, self.quantity, self.targetList)

    def semi_previous_button(self):
        if self.startId <= 0:
            self.startId = 0
            return

        if self.targetList is None:
            self.add_widget(Label(text="No Images"))
            return
        else:
            self.clear_all()
            if self.startId != 0 or self.startId > 0:
                self.startId -= self.quantity
            # print(self.startId)
            self.show_list(self.startId, self.quantity, self.targetList)

    def show_list(self, startId, quantity, targetList):
        print("calling show_list")
        if startId < 0:
            startId = 0
            return
        self.tile = MySmartTile
        if targetList is None:
            print("list is None")
            self.add_widget(Label(text="No Images"))
        else:
            self.len = len(targetList)
            endIndex = min(startId + quantity, self.len)
            for i in range(startId, endIndex):
                self.add_widget(
                    self.tile(
                        parent=self,
                        myid="{}".format(i),
                        targetPath=targetList[i],
                        source=targetList[i],
                    )
                )
            if startId + quantity >= self.len:
                self.add_widget(
                    Label(
                        text="END", 
                        color="white",
                        ))

    def show_node(self, startId, quantity, nodeNmb, clustering=False):
        print(f"nodeNmb is {nodeNmb}")
        self.nodeList = [x for x in self.jsons.keys()]
        self.tiles = []
        
        # modify this path later
        if startId < 0:
            startId = 0
            return

        self.tile = MySmartTile
        if clustering:
            print("On Clustering")
            files = self.jsons[str(self.nodeList[nodeNmb])]
            self.index = nodeNmb
            self.len = len(files)
            print(f"the No of files is {self.len}")
            endIndex = min(startId + quantity, self.len)
            self.updateLabelText()
            for i in range(startId, endIndex):
                try:
                    files[i]
                except IndexError:
                    return
                else:
                    self.add_widget(
                        self.tile(
                            parent=self,
                            myid=f"{i}",
                            target=f"{self.nodeList[nodeNmb]}",
                            targetPath=files[i],
                            source=files[i],
                        )
                    )
        else:
            print("On not Clustering")
            # get indice list of classified images
            self.index = self.jsons[self.nodeList[nodeNmb]]
            # print("indiceList", indiceList)
            self.updateLabelText()
            if len(self.fileList) < 1:
                print("list is None")
                self.add_widget(Label(text="No Images"))
            else:
                nodeList = self.nodeList
                self.len = len(nodeList)
                # print(nodeList)
                endIndex = min(startId + quantity, self.len)
                for i in range(startId, endIndex):
                    # print(i)
                    try:
                        self.fileList[nodeList[i][0]]
                    except IndexError:
                        logging.CRITICAL(
                            f"IndexError index {nodeList[i][0]} in self.fileList"
                        )
                        print("self.fileList: \n", self.fileList)
                        return
                    else:
                        self.add_widget(
                            self.tile(
                                parent=self,
                                myid="{}".format(i),
                                target="{}".format(nodeList[i][1]),
                                targetPath=self.fileList[nodeList[i][0]],
                                source=os.path.join(
                                    '/dataset', self.fileList[nodeList[i][0]]
                                ),
                            )
                        )
                    #  print(f"Tile added\nTarget: {nodeList[i][1]}")
        if startId + quantity >= self.len:
            self.add_widget(Label(
                text="END",
                color="white",
                ))

    def clear_all(self):
        self.clear_widgets()

    def page_next(self):
        self.modeRemain = False
        logging.info("Calling page-next button")
        if self.startId > self.len:
            self.startId = self.len
            return

        if len(self.fileList) < 1:
            self.add_widget(Label(text="No Images"))
        else:
            self.clear_all()
            self.startId += self.quantity
            # print(self.startId)
            self.show_node(
                self.startId, self.quantity, self.index, clustering=self.clustering
            )

    def page_previous(self):
        self.modeRemain = False
        if self.startId <= 0:
            self.startId = 0
            return

        if len(self.fileList) < 1:
            self.add_widget(Label(text="No Images"))
            return
        else:
            self.clear_all()
            if self.startId != 0 or self.startId > 0:
                self.startId -= self.quantity
            # print(self.startId)
            self.show_node(
                self.startId, self.quantity, self.index, clustering=self.clustering
            )

    def updateLabelText(self):
        self.root.ids.node_name.text = f"Current: {self.nodeList[self.index]}"

    def node_next(self):
        self.modeRemain = False
        if self.index < self.mapLength - 1:
            self.index += 1
            self.startId = 0
            print("NODE NUMBER IS ", self.nodeList[self.nodeNmb])
            self.clear_all()
            self.show_node(
                self.startId, self.quantity, self.index, clustering=self.clustering
            )
            self.updateLabelText()
            self.pressButtonList.clear()

    def node_previous(self):
        self.modeRemain = False
        if self.index >= 1:
            self.index -= 1
            self.startId = 0
            self.clear_all()
            self.show_node(
                self.startId, self.quantity, self.index, clustering=self.clustering
            )
            self.updateLabelText()
            self.pressButtonList.clear()

    def getFileList(self, path, quantity):
        list = os.listdir(path=path)[:quantity]
        return list
    
    def update_file_list(self, new_list):
        if new_list:
            self.jsons = new_list

    def save(self):
        
        if self.semiBool:
            # dump json file which has a file path user selected
            if self.root.ids.class_field.text:
                # semi-supervised learning section
                self.write_selected_file(
                    self.root.ids.class_field.text, self.selectFilePathList
                )
                # add a label to label spinner
                self.root.ids.label_spinner.values.append(
                    self.root.ids.class_field.text)
                self.root.ids.label_spinner.values.set()
                print(f"current labels {self.root.ids.label_spinner.values}")
                self.root.ids.label_spinner.text = "Labels"
                
                logging.info("saved json")
                
                # add a label to label spinner
                print(self.root.ids)
                values = self.root.ids.label_spinner.values
                values.append(self.root.ids.class_field.text)
                values = list(set(values))
                self.root.ids.label_spinner.values = values
                print(f"current labels {self.root.ids.label_spinner.values}")
                self.root.ids.label_spinner.text = "Labels"
                
                # initialze text field
                self.root.ids.class_field.text = ""
            else:
                logging.warning("text field is None")
                return

            # Delete select image buttons
            if hasattr(self, "tile"):
                self.rm_selected()
            else:
                logging.warning("plz select images")

        if self.selectSave:
            logging.info("Mode Saving Selected Images")
            if self.root.ids.class_field.text:
                # semi-supervised learning section
                self.write_selected_file(
                    self.root.ids.class_field.text,
                    self.selectFilePathList,
                    "self-labels.json",
                )
                logging.info("saved json")
                
                # add a label to label spinner
                print(self.root.ids)
                values = self.root.ids.label_spinner.values
                values.append(self.root.ids.class_field.text)
                values = list(set(values))
                self.root.ids.label_spinner.values = values
                print(f"current labels {self.root.ids.label_spinner.values}")
                self.root.ids.label_spinner.text = "Labels"
                
                # initialze text field
                self.root.ids.class_field.text = ""
                
            else:
                logging.warning("text field is None")
                return

            # Delete select image buttons
            if hasattr(self, "tile"):
                self.rm_selected()
            else:
                logging.warning("plz select images")

        else:
            logging.info("Mode Saving Unselected Images")
            # dump json file which has a file path user selected
            if self.root.ids.class_field.text:

                self.writeJson(self.root.ids.class_field.text, "cluser_map.json")
                logging.info("saved json")
                
                # add a label to label spinner
                print(self.root.ids)
                values = self.root.ids.label_spinner.values
                values.append(self.root.ids.class_field.text)
                values = list(set(values))
                self.root.ids.label_spinner.values = values
                print(f"current labels {self.root.ids.label_spinner.values}")
                self.root.ids.label_spinner.text = "Labels"
                
                # initialze text field
                self.root.ids.class_field.text = ""
            else:
                logging.warning("text field is None")
                return

            # Delete select image buttons
            if hasattr(self, "tile"):
                self.rm_selected()
            else:
                logging.warning("plz select images")
        self.selectFilePathList = []

    def rm_selected(self):
        logging.debug(self.pressButtonList)
        # get this class's children tiles in a page
        logging.debug(f"tiles\n{self.children}")
        for v in self.pressButtonList:
            logging.debug(f"deleted {v}")
            self.remove_widget(v)
        self.pressButtonList = []

    def bool_noItems(self):
        if len(self.children) == 1:
            logging.info("Annotation finished!")
            return True
        return False

    def openJsonImages(self, json_path):
        path = json_path
        with open(path, "r+") as f:
            jsons = json.load(f)
        # make a DropItemList
        self.mapLength = len(jsons)
        # logging.debug("The No. of Images is ", self.mapLength)
        return jsons

    def writeJson(self, classText, fileName):
        global SAVEDIR
        logging.info("called writeJson")
        saveDir = SAVEDIR
        path = os.path.join(saveDir, fileName)
        # process when pressing Remain button
        print(f"self.modeRemain: {self.modeRemain}, self.selectSave: {self.selectSave}")
        if not self.modeRemain and not self.selectSave:
            for p in self.selectFilePathList:
                self.tilesRemain.append(p)
            print("self.tilesRemain appended")
        else:
            # in remain mode
            for p in self.selectFilePathList:
                if p in self.tilesRemain:
                    self.tilesRemain.remove(p)

        # load existed json file
        if os.path.exists(path):
            logging.info("Loading Json")
            with open(path) as f:
                jsons = json.load(f)
            # update json file
            for v in self.tiles:
                if classText in jsons.keys():
                    jsons[classText].append(v)
                else:
                    add = []
                    add.append(v)
                    jsons[classText] = add
                set(jsons[classText])

        else:
            # write json file
            jsons = {}
            jsons[classText] = self.tiles

        with open(path, "w") as f:
            json.dump(jsons, f, indent=4)
        logging.info("wrote new json")
        
        self.tiles = []

    def extract_unselected_imageLabel(self, pressButtonList, fileList):
        pass

    def write_selected_file(self, label, targetList, fileName="semi-labels.json"):
        global SAVEDIR
        logging.info("called write_selected_file()")

        # TODO delete selected images in original cluster_map.json

        if not os.path.isfile(os.path.join(SAVEDIR, fileName)):
            originJson = {}
            originJson[label] = targetList

        else:
            with open(os.path.join(SAVEDIR, fileName)) as f:
                originJson = json.load(f)
            if label in originJson.keys():
                for target in targetList:
                    originJson[label].append(target) # better using concat?
                    originJson[label] = list(set(originJson[label]))
            else:
                originJson[label] = list(set(targetList))

        #TODO add save remained images in self.tileRemains
        for p in targetList:
            if p in self.tiles:
                self.tiles.remove(p)
                
            if self.modeRemain:
                if p in self.tilesRemain:
                    self.tilesRemain.remove(p)
                
        print(f"self.tiles has {len(self.tiles)} files")

        # merge jsons
        self.jsons.update(originJson)
        
        with open(os.path.join(SAVEDIR, fileName), "w") as f:
            json.dump(originJson, f, indent=4)

        # update self.jsons to reload 
        self.jsons[str(self.nodeList[self.index])].clear()
        for v in self.tiles:
            self.jsons[str(self.nodeList[self.index])].append(v)
            
        targetList.clear()
        logging.info("dump json file of selected files")

    def random_extract(self, percent, targetList):
        pValue = percent * 0.01
        logging.info(f"{percent}% random label extraction start")
        random_sample = random.sample(targetList, math.ceil(len(targetList) * pValue))
        outList = []
        for file in random_sample:
            outList.append(file)
        logging.info(f"Extracted {len(outList)} Images")
        return outList

    def show_openDialog(self):
        import tkinter as tk
        global DIALOG_DEFAULT_PATH
        typ = [('JSON File', '*.json'), ('All', '*')]
        path = DIALOG_DEFAULT_PATH
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename(filetypes=typ, initialdir=path)
        print(file)
        return file
    
    def show_saveDialog(self):
        global DIALOG_DEFAULT_PATH
        typ = [('JSON File', '*.json'), ('All', '*')]
        path = DIALOG_DEFAULT_PATH
        file = filedialog.asksaveasfilename(filetypes=typ, initialdir=path)
        print(file)
        return file
    
    def openFile(self, json_path: str=None, dialog=True):
        global DIALOG_DEFAULT_PATH
        if json_path:
            self.jsons = self.openJsonImages(json_path)
        # add images on the list
        if dialog:
            self.json_path: str = utils.openDialog()
            if len(self.json_path) <= 1:
                print("Please Select JSON file.")
                return False
            
            else:
                _, ext = os.path.splitext(self.json_path)
                print(ext)
                if not ext == '.json':
                    print("This file is not JSON. Please open a JSON file.")
                    return False
                
            # check validity of image path
            datasetRoot = self.checkPathValidity(self.json_path)
            if not datasetRoot:
                return False
            
            # load a json file as dict
            self.jsons = self.openJsonImages(self.json_path)
            
        elif self.jsons:
            datasetRoot = self.checkPathValidity(json_path)
            self.mapLength = len(self.jsons)

        else:
            print("No json file")
            return False
        

        
        # set SmSL mode off
        self.semiBool = False

        # load images
        self.nodeNmb = 0
        print(f"Load map keys {self.jsons.keys()}")
        self.nodeList = [x for x in self.jsons.keys()]
        # add rest of images not selected
        self.nodeList.append("rest")
        
        # torch data loader's shuffle must be false
        self.fileList = []
        for fd_path, sb_fd, sb_f in os.walk(datasetRoot):
            for imageFile in sb_f:
                path = os.path.join(fd_path, imageFile)
                self.fileList.append(path)
        if len(self.fileList) <= 1:
            print(f"there is no file in fileList.")
            self.pop = pop = PopupRaiseError(
            title='Raise Error',
            text="No images found",
            size_hint=(0.4, 0.3),
            pos_hint={'x':0.3, 'y':0.35},
                )
        print(f"first files at {self.fileList[0]}")

        return True
    
    def checkPathValidity(self, jsonPath):
        if not jsonPath:
            print(f"There is not a json file at {jsonPath}")
        
        else:
            
            _, ext = os.path.splitext(jsonPath)
            if not ext == '.json':
                raise ValueError(f"This file is not json: {jsonPath}")
            
            else:

                j = self.openJsonImages(jsonPath)
                k = list(j.keys())[0]
                checkPath = j[k][0]
                print(f"check a file at {checkPath}")
                if not os.path.isfile(checkPath):
                    print(f"There is no file at {checkPath}")
                
                else:
                    datasetRoot = os.path.split(checkPath)[0]
                    return datasetRoot
            
        return False
    
    def showNodeHandler(self, res):
        print(res)

        self.clear_all()
        # load image list from json file made with clustering
        success = self.openFile(json_path="/database/cluster_map.json", dialog=False)
        if success:
            self.show_node(
                self.startId, self.quantity, self.index, clustering=self.clustering
            )
        else:
            pass

    def on_labelSpinner(self, instance, text):
        
        self.root.ids.class_field.text = text
            
    
    
    
    
class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    
class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

    
            
