from kivymd.app import MDApp
from kivy.core.window import Window
from widgets.MainScreen import MainScreen
from widgets.Controller import Controller # load before bulding.
from kivy.properties import ListProperty
from kivy.utils import rgba
import asyncio
Window.size = (1400, 800)


class TestApp(MDApp):
    bg_color = ListProperty(rgba(0.3, 0.3, 0.3, 1))
    outline_color = ListProperty(rgba(0.8, 0.8, 0.8, 1))
    
    def build(self):
        main = MainScreen()
        return main
    
def main():
    app = TestApp()
    app.run()
    
if __name__ == "__main__":
    main()
