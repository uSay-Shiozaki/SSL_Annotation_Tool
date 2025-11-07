import os
import sys
import json
from kivy.event import EventDispatcher
from kivy.core.window import Window
from kivy.properties import ObjectProperty

class Controller(EventDispatcher):
    """Class vars here"""
    is_remain_mode = False
    is_smsl_mode = False
    root = ObjectProperty(None)
        
    def on_kv_post(self, base_widget):
        # Binding Enter Key
        self.save_button = self.root.ids.save
        Window.bind(on_key_down=self._on_enter_down)
 
    def _on_enter_down(self, window, key, scancode, codepoint, modifier):
        if key in (13, 271): # 13: Enter, 271: Num Pad Enter
            self.save_button.trigger_action(duration=0.1)
        
    def autosave(self, file_dir, target, remain_tiles):
        target_json = target
        assert target_json

        tiles_in_remain = remain_tiles

        if 'remain' not in target_json.keys():
            target_json['remain'] = []
 
        for tile in tiles_in_remain:
            if (tile.cluster in target_json.keys()) and (tile.targetPath in target_json[tile.cluster]):
                target_json[tile.cluster].remove(tile.targetPath)
                
        target_json['remain'] = [tile.targetPath for tile in tiles_in_remain]

        filtered = {k: v for k, v in target_json.items() if len(v) != 0}
        with open(file_dir + '/autosave.json', 'w') as f:
            json.dump(filtered, f, indent=4)

    def is_remain_mode(self):
        return True if self.is_remain_mode else False

    def is_smsl_mode(self):
        return True if self.is_smsl_mode else False

    def switch_mode(self, target):
        target = not target

    
