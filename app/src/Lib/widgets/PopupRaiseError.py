from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.spinner import Spinner
from kivy.uix.button import Button
from kivy.uix.label import Label
import utils

class PopupRaiseError(Popup):
  __events__ = ('on_yes','on_no')
  message = StringProperty('')
  
  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    self.auto_dismiss = False
    
  def on_yes(self):
    pass
  
  def on_no(self):
    pass
  