from kivymd.app import MDApp
from kivy.core.window import Window
from widgets.MainScreen import MainScreen
from kivy.network.urlrequest import UrlRequest
Window.size = (1920, 720)


class TestApp(MDApp):
    def build(self):
        main = MainScreen()
        return main

    def on_success(self):
        print("Posted successfully")

    def on_failure(self):
        print("Failed Posting")

    def on_error(self):
        print("Error occured!")

    def on_progress(self):
        print("On progress")

    def getClusteringTable(self):
        endPoint: str = 'http://ssl_network:80/api/clustering'
        req = UrlRequest(endPoint, on_success=self.on_success, 
                         on_failure=self.on_failure, on_error=self.on_error,
                         on_progress=self.on_progress)
        
        return req


if __name__ == "__main__":

    app = TestApp()
    app.run()
