from kivymd.app import MDApp
from kivy.core.window import Window
from widgets.MainScreen import MainScreen
Window.size = (1920, 720)


class TestApp(MDApp):
    def build(self):
        main = MainScreen()
        return main


if __name__ == "__main__":

    app = TestApp()
    app.run()
