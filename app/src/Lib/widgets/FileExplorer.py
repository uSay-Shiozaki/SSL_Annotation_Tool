from kivy.app import App
from kivy.lang import Builder

kv = Builder.load_string("""
Screen:
  BoxLayout:
      orientation:'vertical'
      Label:
          text:'File Explorer'
      ToggleButton:
          id: btnCloseHandler
          text:'Close'
          on_release: app.stop()
                         """)

class FileExplorerScreen(App):
    
    def build(self):
        return kv
    
if __name__ == '__main__':
    print('File Explorer has been Called')
    FileExplorerScreen().run()