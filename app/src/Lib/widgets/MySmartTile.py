from kivymd.uix.imagelist import MDSmartTile
import logging
from kivy.graphics import Color, Rectangle


class MySmartTile(MDSmartTile):
    def __init__(self, parent, myid, target=None, targetPath=None, **kwargs):
        super().__init__(**kwargs)
        self.myparent = parent
        self.id = myid
        self.target = target
        self.targetPath = targetPath
        self.press = True
        self.myparent.tiles.append(self.targetPath)
        self.mipmap = True

    def on_release(self):
        self.myparent.tileId = self.id
        self.myparent.target = self.target
        logging.info(
            f"\nThis tileId is {self.myparent.tileId}\nTarget is {self.myparent.target}\nFlie Path is {self.targetPath}"
        )
        if self.press:
            with self.canvas.after:
                if not self.myparent.selectSave:
                    self.color = Color(0,0,0,1)
                else:
                    self.color = Color(1,0,0,0.8)
                self.rect = Rectangle(pos=self.pos, size=self.size)
                
            if not self.myparent.selectSave:
                # save button instance to manipulate button tiles
                if not self in self.myparent.pressButtonList:
                    self.myparent.pressButtonList.append(self)
                    
                # save target image path to create json map
                if not self.targetPath in self.myparent.selectFilePathList:
                    self.myparent.selectFilePathList.append(self.targetPath)
            else:
                if not self in self.myparent.pressButtonList:
                    self.myparent.selectFilePathList.append(self.targetPath)
                    self.myparent.pressButtonList.append(self)
                    self.myparent.tiles.remove(self.targetPath)
            self.press = False
        else:
            self.canvas.after.remove(self.color)
            self.canvas.after.remove(self.rect)

            if not self.myparent.selectSave:
                if self in self.myparent.pressButtonList:
                    self.myparent.pressButtonList.remove(self)

                if self.targetPath in self.myparent.selectFilePathList:
                    self.myparent.selectFilePathList.remove(self.targetPath)
            else:
                if self in self.myparent.pressButtonList:
                    self.myparent.selectFilePathList.remove(self.targetPath)
                    self.myparent.pressButtonList.remove(self)
                    
                    try:
                        self.myparent.tiles.append(self.targetPath)
                    except FileNotFoundError:
                        print(f"Removing Failed at {self.targetPath}")

            self.press = True

        # Updating Preview Box 
        rect_canvas = self.myparent.root.ids.preview_box.canvas.get_group('rect')[0] 
        rect_canvas.source = self.targetPath
        logging.debug(self.myparent.pressButtonList)
        logging.debug(self.myparent.selectFilePathList)
