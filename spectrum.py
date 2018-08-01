"""
Based on https://gist.github.com/ZWMiller/53232427efc5088007cab6feee7c6e4c
"""
import threading
from math import ceil
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from pyfirmata import Arduino
from scipy.signal import butter, sosfiltfilt, decimate

import led_matrix

global keep_going


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='bandpass', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def discretize_plot(data, xbins, ybins, maxval):
    downsample = decimate(data, int(ceil(len(data) / xbins)), zero_phase=True)
    return [int((val / maxval) * ybins) for val in downsample]


class SpectrumPlotter:
    def __init__(self):
        self.init_plot()
        self.init_mic()
        self.init_matrix()

        self.annotation_list = []

    def init_plot(self):
        print('Initializing plot...')
        _, self.ax = plt.subplots(3)

        # Prepare the Plotting Environment with random starting values
        x1 = np.arange(10000)
        y1 = np.random.randn(10000)
        x2 = np.arange(8)
        y2 = np.random.randn(8)

        # Plot 0 is for raw audio data
        self.li, = self.ax[0].plot(x1, y1)
        self.ax[0].set_xlim(0, 1000)
        self.ax[0].set_ylim(-5000, 5000)
        self.ax[0].set_title("Raw Audio Signal")
        # Plot 1 is for the FFT of the audio
        self.li2, = self.ax[1].plot(x1, y1)
        self.ax[1].set_xlim(0, 2000)
        self.ax[1].set_ylim(0, 1000000)
        self.ax[1].set_title("Fast Fourier Transform")
        # Plot 2 is for the binned FFT
        self.li3 = self.ax[2].plot(x2, y2, 'ro')[0]  # for some reason, returned as a list of 1
        self.ax[2].set_xlim(0, 7)
        self.ax[2].set_ylim(0, 7)
        self.ax[2].set_title("8-Binned FFT")
        # Show the plot, but without blocking updates
        plt.pause(0.01)
        plt.tight_layout()
        print('Done')

    def init_mic(self):
        print('Initializing mic...')
        FORMAT = pyaudio.paInt16  # We use 16bit format per sample
        CHANNELS = 1
        self.RATE = 44100 // 2
        self.CHUNK = 1024  # 1024bytes of data red from a buffer

        self.audio = pyaudio.PyAudio()

        # start Recording
        self.stream = self.audio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=self.RATE,
                                      input=True)  # ,
        # frames_per_buffer=CHUNK)

        global keep_going
        keep_going = True
        print('Done')

    def init_matrix(self):
        print('Initializing Arduino...')
        self.board = Arduino('COM3')
        print('Done')
        self.matrix = led_matrix.LedMatrix(self.board)
        self.matrix.setup()

    def process_data(self, in_data):
        # get and convert the data to float
        audio_data = np.fromstring(in_data, np.int16)
        # apply band-pass to amplify human speech range
        audio_data = butter_bandpass_filter(audio_data, 300, 3400, self.RATE, 20)
        # Fast Fourier Transform, 10*abs to scale it up and make sure it's all positive
        dfft = 10*abs(np.fft.rfft(audio_data))[:300]

        # Force the new data into the plot, but without redrawing axes.
        # If uses plt.draw(), axes are re-drawn every time
        self.li.set_xdata(np.arange(len(audio_data)))
        self.li.set_ydata(audio_data)
        self.li2.set_xdata(np.arange(len(dfft)) * 10.)
        self.li2.set_ydata(dfft)
        self.li3.set_xdata(np.arange(8))
        self.li3.set_ydata(discretize_plot(dfft, 8, 8, 1000000))

        for a in self.annotation_list:
            a.remove()
            self.annotation_list.remove(a)
        for i, txt in enumerate(self.li3.get_ydata()):
            self.annotation_list.append(
                self.ax[2].annotate(str(txt), (self.li3.get_xdata()[i], self.li3.get_ydata()[i])))

        # Show the updated plot, but without blocking
        plt.pause(1 / 30)
        return keep_going

    def start_listening(self):
        global keep_going
        # Open the connection and start streaming the data
        self.stream.start_stream()
        print("\n+---------------------------------+")
        print("| Press Ctrl+C to Break Recording |")
        print("+---------------------------------+\n")

        def update_matrix():
            while True:
                for col, val in enumerate(reversed(self.li3.get_ydata())):
                    self.matrix.maxAll(int(col + 1), int((2 * pow(2, val - 1)) - 1))
                sleep(1 / 30)

        threading.Thread(target=update_matrix, daemon=True).start()

        # Loop so program doesn't end while the stream callback's
        # itself for new data
        while keep_going:
            try:
                self.process_data(self.stream.read(self.CHUNK))
            except KeyboardInterrupt:
                keep_going = False

        # Close up shop (currently not used because KeyboardInterrupt
        # is the only way to close)
        self.stream.stop_stream()
        self.stream.close()

        self.audio.terminate()


if __name__ == "__main__":
    p = SpectrumPlotter()
    p.start_listening()
