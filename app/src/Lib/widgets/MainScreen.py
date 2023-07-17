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

    def swavModuleExecution(self):
        boolSemi = False
        boolCreateDataset = False

        if not boolSemi:
            try:
                os.system(
                    "python -m torch.distributed.launch --nproc_per_node=1 \
                        ./ml_models/mycodes/app_modules/swav_modules/swav_semisup_module.py"
                )
            except:
                print("semi-learning failed")
                return
            boolSemi = True

        else:
            try:
                os.system(
                    "python -m torch.distributed.launch --nproc_per_node=1 \
                        ./ml_models/mycodes/app_modules/swav_modules/swav_classification_module.py"
                )
            except:
                print("classification failed")

    def ibotModuleExecution(self):
        try:
            os.system(
                "python -m torch.distributed.launch --nproc_per_node=1 \
                ./mypackages/ibot_unsup_module.py"
            )
        except:
            print("Clustering Failed")
            return
