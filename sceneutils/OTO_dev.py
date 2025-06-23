import time
import numpy as np
from skimage import color
from pathlib import Path

ParentPath = Path(__file__).parent.parent
media_path = ParentPath / 'media'
sound_path = media_path / 'sounds'

def test(instate, outstate):
    name='test'
    buffers=outstate['buffers']

    if instate['count'] == 0:
        buffers.register_generator(name)
        test=1  
        return

    if instate['count'] == -1:
        test=-1
        return
    
    test=instate['count']
