from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.properties import StringProperty
import time


class Timer(Widget):
    # Add property
    text = StringProperty()
    # When clicked

    def on_countDown(self):
        self.text = '3'
        for i in range(3):
            time.sleep(1)
            self.text = str(int(self.text) - 1)
            print(self.text)

    def on_command(self, **kwargs):
        Clock.schedule_interval(self.countUp, 0.1)

    def countUp(self, dt):
        self.text = "{:.3f}".format(float(self.text) + 0.1)

    def countStop(self):
        Clock.unschedule(self.countUp)
        print(self.text)

    def startTimer(self):
        if self.startBool:
            self.on_countDown()
            self.startBool = False
        self.on_command()

    def stopTimer(self):
        self.countStop()

    def __init__(self, **kwargs):
        self.startBool = True
        super().__init__(**kwargs)
