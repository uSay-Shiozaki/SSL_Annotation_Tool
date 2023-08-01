from kivy.uix.spinner import Spinner
from widgets.PopupWaiting import YNPopup, PopupWaiting
import threading
import utils
from kivy.clock import mainthread
class StartSpinner(Spinner):
  root = None
  
  def on_startSpinner(self, instance, text):
      if text == "SSL":
          print("SSL")
          # self.root.ids.image_grid.runSSL()
          self.popup_open('Are you sure runnning SSL?', 'ssl')

      elif text == "SmSL iBOT":
          print("SmSL with iBOT")
          # self.ids.image_grid.runSmSLwithiBOT()
          self.popup_open('Are you sure runnning SmSL with iBOT?', 'smsl-ib')

      elif text == "SmSL SwAV":
          print("SmSL with SwAV")
          # self.ids.image_grid.semi_learning_button()
          self.popup_open('Are you sure runnning SmSL with SwAV', 'smsl-sw')

      elif text == "Load Annotation Data":
          print("Load Annotation Data")
          self.root.ids.image_grid.start()
      
      instance.text = "start"
      

  def popup_open(self, text, mode):
    self.pop = pop = YNPopup(
      title='popup',
      message=text,
      size_hint=(0.4, 0.3),
      pos_hint={'x':0.3, 'y':0.35},
    )
    if mode == 'ssl':
      pop.bind(
        on_yes=self._popup_yes_ssl,
        on_no=self._popup_no,
      )
    elif mode == 'smsl-ib':
      pop.bind(
      on_yes=self._popup_yes_smsl_ib,
      on_no=self._popup_no,
    )
    elif mode == 'smsl-sw':
      pop.bind(
        on_yes=self._popup_yes_sw,
        on_no=self._popup_no,
      )
    self.pop.open()
    
  def _popup_yes_ssl(self, instance):
    self.res = None
    self._popup_close()
    self.popup_waiting()
    
    def _process():
      self.res = utils.runSSL()
      self._showNodeHandler()
      self._popup_waiting_close()
      
    self.thread1 = threading.Thread(target=_process, daemon=True)
    self.thread1.start()
    
  @mainthread
  def _popup_close(self):
    self.pop.dismiss()
  
  @mainthread
  def _showNodeHandler(self):
    self.root.ids.image_grid.showNodeHandler(self.res)

  def _popup_yes_smsl_ib(self, instance):
    # self.root.ids.image_grid.start_semisupervised_learning()
    self.pop.dismiss()
    
  def _popup_yes_smsl_sw(self, instance):
    # self.root.ids.image_grid.start_semisupervised_learning()
    self.pop.dismiss()
    
  def _popup_no(self, instance):
    self.pop.dismiss()
    
  def popup_waiting(self):
    self.pop_waiting = pop_waiting = PopupWaiting(
        title='popup',
        message='Running Process ...',
        size_hint=(0.4, 0.3),
        pos_hint={'x':0.3, 'y':0.35},
    )
    
    self.pop_waiting.open()
    
  @mainthread
  def _popup_waiting_close(self):
    self.pop_waiting.dismiss()
  
    
    
  