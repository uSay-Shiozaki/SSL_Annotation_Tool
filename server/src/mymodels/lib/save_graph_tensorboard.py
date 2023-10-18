import numpy as np
import cv2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = 'events.out.tfevents.xxxxxxxxxx.xxxxxxxxxx' # Tensorboard ログファイル
event_acc = EventAccumulator(path, size_guidance={'images': 0})
event_acc.Reload() # ログファイルのサイズによっては非常に時間がかかる

for tag in event_acc.Tags()['images']:
    events = event_acc.Images(tag)
    tag_name = tag.replace('/', '_')
    for index, event in enumerate(events):
        # 画像はエンコードされているので戻す
        s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
        image = cv2.imdecode(s, cv2.IMREAD_COLOR) # カラー画像の場合
        # 保存
        outpath = '{}_{:04}.jpg'.format(tag_name, index)
        cv2.imwrite(outpath, image)