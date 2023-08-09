from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.spinner import Spinner
from kivy.uix.button import Button
from kivy.uix.label import Label
import utils

class PopupArgumentsforSSL(Popup):
  __events__ = ('on_yes', 'on_no')
  message = StringProperty('')
  data_path = ""
  arch = "vit_small"
  
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    self.auto_dismiss = False
    
  def on_yes(self):
    pass
  
  def on_no(self):
    pass
  
class OpenDialogButton(Button):
  parent = None

  def on_release(self, **kwargs):
    super().__init__(**kwargs)
    path = utils.openDirDialog()
    if path:
      self.parent.ids.input_data_path.text = path
  
class WarningLabel(Label):
  parent = None
  