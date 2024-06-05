from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.properties import StringProperty
import time


class Timer(Widget):
    # Add property
    text = StringProperty()
    # When clicked

    def __init__(self, **kwargs):
        self.startBool = True
        super().__init__(**kwargs)

    def _countDown(self, event):
        self.text = str(int(self.text) - 1)
        print(self.text)
            
    def on_countDown(self, event):
        self.text = "4"
        Clock.schedule_interval(self._countDown, 1.0)
        if int(self.text) == 0:
            self.startBool = False
            
    def on_command(self, **kwargs):
        if self.startBool:
            Clock.schedule_interval(self.on_countDown, 1.0)
        else:
            Clock.schedule_interval(self.countUp, 0.1)

    def countUp(self, dt):
        self.text = "{:.3f}".format(float(self.text) + 0.1)

    def countStop(self):
        Clock.unschedule(self.countUp)
        print(self.text)

    def stopTimer(self):
        self.countStop()
        
    def reset(self):
        self.text = "0.0"
        self.startBool = True


