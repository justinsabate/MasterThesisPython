import cv2
import numpy as np
from IPython.display import Audio
import librosa
face_cascade = cv2.CascadeClassifier('/Users/justinsabate/ThesisPython/virtualenv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
option1=1

'''audio file reading'''
signal = 'test.wav'
y, sr = librosa.load('../../wavfiles/1st violin.wav')

Audio(data=y, rate=sr)


while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if option1:
        faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            print(x) # from 300 (right) to 600 (left) for approx 45° rotation

        # cv2.imshow('Face Recognition', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()


'''audio things to be included with head detection '''
''' see https://python-sounddevice.readthedocs.io/en/0.4.5/examples.html for examples to create the sound stream '''

# #!/usr/bin/env python3
# """Creating an asyncio generator for blocks of audio data.
#
# This example shows how a generator can be used to analyze audio input blocks.
# In addition, it shows how a generator can be created that yields not only input
# blocks but also output blocks where audio data can be written to.
#
# You need Python 3.7 or newer to run this.
#
# """
# import asyncio
# import queue
# import sys
#
# import numpy as np
# import sounddevice as sd
#
#
# async def inputstream_generator(channels=1, **kwargs):
#     """Generator that yields blocks of input data as NumPy arrays."""
#     q_in = asyncio.Queue()
#     loop = asyncio.get_event_loop()
#
#     def callback(indata, frame_count, time_info, status):
#         loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
#
#     stream = sd.InputStream(callback=callback, channels=channels, **kwargs)
#     with stream:
#         while True:
#             indata, status = await q_in.get()
#             yield indata, status
#
#
# async def stream_generator(blocksize, *, channels=1, dtype='float32',
#                            pre_fill_blocks=10, **kwargs):
#     """Generator that yields blocks of input/output data as NumPy arrays.
#
#     The output blocks are uninitialized and have to be filled with
#     appropriate audio signals.
#
#     """
#     assert blocksize != 0
#     q_in = asyncio.Queue()
#     q_out = queue.Queue()
#     loop = asyncio.get_event_loop()
#
#     def callback(indata, outdata, frame_count, time_info, status):
#         loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))
#         outdata[:] = q_out.get_nowait()
#
#     # pre-fill output queue
#     for _ in range(pre_fill_blocks):
#         q_out.put(np.zeros((blocksize, channels), dtype=dtype))
#
#     stream = sd.Stream(blocksize=blocksize, callback=callback, dtype=dtype,
#                        channels=channels, **kwargs)
#     with stream:
#         while True:
#             indata, status = await q_in.get()
#             outdata = np.empty((blocksize, channels), dtype=dtype)
#             yield indata, outdata, status
#             q_out.put_nowait(outdata)
#
#
# async def print_input_infos(**kwargs):
#     """Show minimum and maximum value of each incoming audio block."""
#     async for indata, status in inputstream_generator(**kwargs):
#         if status:
#             print(status)
#         print('min:', indata.min(), '\t', 'max:', indata.max())
#
#
# async def wire_coro(**kwargs):
#     """Create a connection between audio inputs and outputs.
#
#     Asynchronously iterates over a stream generator and for each block
#     simply copies the input data into the output block.
#
#     """
#     async for indata, outdata, status in stream_generator(**kwargs):
#         if status:
#             print(status)
#         outdata[:] = indata
#
#
# async def main(**kwargs):
#     print('Some informations about the input signal:')
#     try:
#         await asyncio.wait_for(print_input_infos(), timeout=2)
#     except asyncio.TimeoutError:
#         pass
#     print('\nEnough of that, activating wire ...\n')
#     audio_task = asyncio.create_task(wire_coro(**kwargs))
#     for i in range(10, 0, -1):
#         print(i)
#         await asyncio.sleep(1)
#     audio_task.cancel()
#     try:
#         await audio_task
#     except asyncio.CancelledError:
#         print('\nwire was cancelled')
#
#
# if __name__ == "__main__":
#     try:
#         asyncio.run(main(blocksize=1024))
#     except KeyboardInterrupt:
#         sys.exit('\nInterrupted by user')