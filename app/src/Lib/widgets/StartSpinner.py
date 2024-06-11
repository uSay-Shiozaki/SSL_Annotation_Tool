from kivy.uix.spinner import Spinner
from widgets.PopupWaiting import YNPopup, PopupWaiting
from widgets.PopupArguments import PopupArgumentsforSSL, PopupArgumentsforClust
import threading
import utils
from kivy.clock import mainthread, Clock

class StartSpinner(Spinner):
    root = None

    def on_startSpinner(self, instance, text):
        if text == "Clust":
            print("Clust")
            # self.root.ids.image_grid.runClust()
            self.popup_open("start clustering?", 'clust')

        if text == "SSL":
            print("SSL")
            # self.root.ids.image_grid.runSSL()
            self.popup_open('run SSL?', 'ssl')

        elif text == "SmSL iBOT":
            print("SmSL with iBOT")
            # self.ids.image_grid.runSmSLwithiBOT()
            self.popup_open('run SmSL with iBOT?', 'smsl-ib')

        elif text == "SmSL SwAV":
            print("SmSL with SwAV")
            # self.ids.image_grid.semi_learning_button()
            self.popup_open('run SmSL with SwAV', 'smsl-sw')

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
        if mode == 'clust':
            self.pop = pop = PopupArgumentsforClust(
        title='Arguments',
        size_hint=(0.4, 0.3),
        pos_hint={'x': 0.3, 'y': 0.35}
      )
            pop.bind(
        on_yes=self._popup_yes_clust,
        on_no=self._popup_no,
      )
        if mode == 'ssl':
            self.pop = pop = PopupArgumentsforSSL(
        title='Arguments',
        size_hint=(0.4, 0.3),
        pos_hint={'x':0.3, 'y':0.35},
        )
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
        on_yes=self._popup_yes_smsl_sw,
        on_no=self._popup_no,
      )
        self.pop.open()

    def _popup_yes_clust(self, instance):
        self.res = None
        self.json = {
      "data_path": self.pop.ids.input_data_path_clust.text,
      "pretrained_weights": self.pop.ids.input_weight_path_clust.text,
      "arch": self.pop.ids.arch_spinner.text
    }
        if len(self.json["data_path"]) < 1 or self.json == "Model Size":

            def do(dt):
                self.pop.ids.warning.text = "please fill all arguments"

            Clock.schedule_once(do)

        else:

            def _process():
                print(self.json)
                self.res = utils.runClustering(self.json)
                self._showNodeHandler()
                self._popup_waiting_close()

            self._popup_close()
            self.popup_waiting()
            self.thread1 = threading.Thread(target=_process, daemon=True)
            self.thread1.start()

    def _popup_yes_ssl(self, instance):
        self.res = None
        self.json = {
            "data_path": self.pop.ids.input_data_path.text,
            "arch": self.pop.ids.arch_spinner.text,
        }

        if len(self.json['data_path']) < 1 or self.json == "Model Size":
            def do(dt):
                self.pop.ids.warning.text = "please fill all arguments"
            Clock.schedule_once(do)

        else:
            def _process():
                print(self.json)
                self.res = utils.runSSL(self.json)
                if self.res != None:
                  self._showNodeHandler()
                  self._popup_waiting_close()

            self._popup_close()
            self.popup_waiting()
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
