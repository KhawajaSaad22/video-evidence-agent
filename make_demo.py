"""Generate a codec smoke video; colors/shapes do not benchmark action understanding."""
from pathlib import Path
import cv2
import numpy as np

def main():
    root=Path('data'); root.mkdir(exist_ok=True); path=root/'demo.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,240))
    if not writer.isOpened(): raise RuntimeError('MJPG encoder unavailable')
    try:
        for i in range(120):
            frame=np.full((240,320,3),245,dtype='uint8')
            if i<40: cv2.rectangle(frame,(60,50),(240,190),(0,0,240),-1)
            elif i<80: cv2.circle(frame,(160,120),65,(240,0,0),-1)
            else: cv2.rectangle(frame,(60,50),(240,190),(0,190,0),-1)
            writer.write(frame)
    finally: writer.release()
    print(path)
if __name__=='__main__': main()
