from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty

class PopupMenu(BoxLayout):
  popup_close = ObjectProperty(None)
  
class YNPopup(Popup):
  __events__ = ('on_yes', 'on_no')
  message = StringProperty('')
  
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    self.auto_dismiss = False
    
  def on_yes(self):
    pass
  
  def on_no(self):
    pass

class PopupWaiting(Popup):
  message = StringProperty('')
  
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    self.auto_dismiss = False
