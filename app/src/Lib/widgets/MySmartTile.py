from kivymd.uix.imagelist import MDSmartTile
import logging


class MySmartTile(MDSmartTile):
    def __init__(self, parent, myid, target=None, targetPath=None, **kwargs):
        super().__init__(**kwargs)
        self.myparent = parent
        self.id = myid
        self.target = target
        self.targetPath = targetPath
        self.press = True
        self.myparent.tiles.append(self.targetPath)

    def on_release(self):
        self.myparent.tileId = self.id
        self.myparent.target = self.target
        logging.info(
            f"\nThis tileId is {self.myparent.tileId}\nTarget is {self.myparent.target}\nFlie Path is {self.targetPath}"
        )
        if self.press:
            self.lines = 2
            self.box_color = (0, 0, 0, 0.9)
            if not self.myparent.selectSave:
                # save buttonID for new json file
                if not int(self.id) in self.myparent.pressButtonList:
                    self.myparent.pressButtonList.append(int(self.id))

                if not self.targetPath in self.myparent.selectFilePathList:
                    self.myparent.selectFilePathList.append(self.targetPath)
            else:
                if not self.id in self.myparent.pressButtonList:
                    self.myparent.selectFilePathList.append(self.targetPath)
                    self.myparent.pressButtonList.append(int(self.id))
                    self.myparent.tiles.remove(self.targetPath)
            self.press = False
        else:
            self.box_color = (0, 0, 0, 0)
            if not self.myparent.selectSave:
                if int(self.id) in self.myparent.pressButtonList:
                    self.myparent.pressButtonList.remove(int(self.id))

                if self.targetPath in self.myparent.selectFilePathList:
                    self.myparent.selectFilePathList.remove(self.targetPath)
            else:
                if int(self.id) in self.myparent.pressButtonList:
                    self.myparent.selectFilePathList.remove(self.targetPath)
                    self.myparent.pressButtonList.remove(int(self.id))
                    
                    try:
                        self.myparent.tiles.append(self.targetPath)
                    except FileNotFoundError:
                        print(f"Removing Failed at {self.targetPath}")

            self.press = True
        self.myparent.previewSrc = self.targetPath
        logging.debug(self.myparent.pressButtonList)
        logging.debug(self.myparent.selectFilePathList)
