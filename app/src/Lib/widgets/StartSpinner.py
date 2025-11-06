from kivy.uix.spinner import Spinner
from widgets.PopupWaiting import YNPopup, PopupWaiting
from widgets.PopupArguments import PopupArgumentsforSSL, PopupArgumentsforClust
import threading
import utils
import os
from kivy.clock import mainthread, Clock

class StartSpinner(Spinner):
    root = None

    def initialize(self):
        if os.path.exists("/database/cluster_map.json"):
            os.remove("/database/cluster_map.json")
        with open("/database/cluster_map.json", mode='w') as f:
            f.write("{}")

        self.root.ids.image_grid.initialize_vars()

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
            # self.popup_open('load json as annotations', 'load-anno')
            self.initialize()


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
        elif mode == 'load-anno':
            pop.bind(
                on_yes=self._popup_yes_load_anno,
                on_no=self._popup_no,
            )
        self.pop.open()

    def _popup_yes_load_anno(self, instance):
        pass

    def _popup_yes_clust(self, instance):
        self.res = None
        self.json = {
      "data_path": self.pop.ids.input_data_path.text,
      "pretrained_weights": self.pop.ids.input_weight_path_clust.text,
      "arch": self.pop.ids.arch_spinner.text,
      "n_clusters": int(self.pop.ids.n_clusters.text) if self.pop.ids.n_clusters.text else 10,
    }
        # initialize arguments if nothing.
        if self.json['arch'] == '':
            self.json['arch'] = 'vit_small'
        
        # validation
        if len(self.json["data_path"]) < 1 or self.json == "Model Size":

            def do(dt):
                self.pop.ids.warning.text = "please fill all arguments"

            Clock.schedule_once(do)

        else:

            def _process():
                print(self.json)
                self.res = utils.runClustering(self.json)
                print("clustering completed")

                self._showNodeHandler()
                self._popup_waiting_close()

            self._popup_close()
            self.popup_waiting()
            self.thread1 = threading.Thread(target=_process)
            self.thread1.start()

            

    def _popup_yes_ssl(self, instance):
        self.res = None
        self.json = {
            "data_path": self.pop.ids.input_data_path.text,
            "arch": self.pop.ids.arch_spinner.text,
            "n_clusters": int(self.pop.ids.n_clusters.text) if self.pop.ids.n_clusters.text else 10,
        }

        if self.json['n_clusters'] == '':
            def do(dt):

                self.pop.ids.warning.text = "please set the number of clusters"
            cclock.schedule_once(do)

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
        print("showing in node handler")

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
