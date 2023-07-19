from tracemalloc import start
from webbrowser import BackgroundBrowser
from kivy.uix.boxlayout import BoxLayout
from widgets.MyGridLayout import MyGridLayout
from kivy.uix.button import Button
import os
import subprocess


class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Main Screen Launched")
