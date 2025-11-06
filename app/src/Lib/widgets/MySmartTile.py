from kivymd.uix.imagelist import MDSmartTile
import logging
from kivy.graphics import Color, Rectangle


class MySmartTile(MDSmartTile):
    def __init__(self, parent, myid, target=None, targetPath=None, cluster=None, **kwargs):
        super().__init__(**kwargs)
        self.myparent = parent
        self.id = myid
        self.target = target
        self.targetPath = targetPath
        self.press = False
        self.myparent.tiles.append(self)
        self.mipmap = True
        self.cluster: str = cluster

    def on_release(self):
        self.myparent.tileId = self.id
        self.myparent.target = self.target
        logging.info(
            f"\nThis tileId is {self.myparent.tileId}\nTarget is {self.myparent.target}\nFlie Path is {self.targetPath}"
        )
        if not self.box_color == (0,0,0,0):
            self.box_color = (0,0,0,0)

        if not self.press:
            if not self.myparent.selectSave:
                with self.canvas.after:
                    self.color = Color(0,0,0,1)
                    self.rect = Rectangle(pos=self.pos, size=self.size)
            else:
                with self.canvas.after:
                    self.color = Color(1,0,0,0.8)
                    self.rect = Rectangle(pos=self.pos, size=self.size)

            if self not in self.myparent.pressButtonList:
                self.myparent.pressButtonList.append(self)

            self.press = True
        else:
            self.canvas.after.remove(self.rect)
            self.canvas.after.remove(self.color)

            if self in self.myparent.pressButtonList:
                self.myparent.pressButtonList.remove(self)
                    
            self.press = False

        # Updating Preview Box 
        rect_canvas = self.myparent.root.ids.preview_box.canvas.get_group('rect')[0] 
        rect_canvas.source = self.targetPath
        logging.debug(self.myparent.pressButtonList)
        logging.debug(self.myparent.selectFilePathList)
