import matplotlib.pyplot as plt
import numpy as np


def plot_HRTF_refinement(freq, allPassFilter_l, allPassFilter_r, HRTF_L, HRTF_R, NFFT, HRTF_refinement_plot):
    if HRTF_refinement_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        positionIndex = 10000
        ax1.plot(freq, 10 * np.log10(abs(allPassFilter_l[positionIndex, 0:NFFT // 2 + 1])))
        ax1.set_xscale('log')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_ylim(-1, 1)
        ax1.set_title('Spectrum')

        ax2.plot(freq, np.unwrap(np.angle(allPassFilter_l[positionIndex, 0:NFFT // 2 + 1])), label="l")
        ax2.plot(freq, np.unwrap(np.angle(allPassFilter_r[positionIndex, 0:NFFT // 2 + 1])), label="r")
        ax2.set_xscale('log')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (rad)')
        ax2.set_title('Phase')
        plt.legend()

        # ax3.plot(freq, np.unwrap(np.angle(HRTF_L[positionIndex])) - np.unwrap(np.angle(HRTF_R[positionIndex])),
        #          label='Difference LR in phases - non filtered')
        # # ax3.plot(freq, np.unwrap(np.angle(HRTF_R[positionIndex])), label='non filtered R')
        ax3.plot(freq, np.unwrap(np.angle(HRTF_L[positionIndex])) - np.unwrap(np.angle(HRTF_R[positionIndex])),
                 label='Difference LR in phases - filtered')
        # ax3.plot(freq, np.unwrap(np.angle(HRTF_R[positionIndex])), label='filtered R')
        ax3.set_xscale('log')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Phase (rad)')
        ax3.set_title('Phase comparison HRTF')
        plt.legend()
        plt.show()


def plot_radial(freq, dn, radial_plot):
    if radial_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)  # plots
        for i in range(0, len(dn)):
            ax1.plot(freq, 20 * np.log10(abs(dn[i])), label='N = ' + str(i))
            ax2.plot(freq, np.unwrap(np.angle(dn[i])), label='N = ' + str(i))
        ax1.set_xscale('log')
        # ax.set_yscale('log')
        ax1.set_xlim(np.min(freq) + 1, np.max(freq))
        # ax.set_ylim(np.min(abs(dn[1])), np.max(abs(dn[1])))
        plt.legend()
        ax1.set_xlabel('Frequency (Hz)')
        ax2.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude (dB)')
        ax2.set_ylabel('Phase (rad)')
        ax1.set_title('Spectrum')
        ax2.set_title('Phase')
        plt.show()


def plot_freq_output(freq, Sl, Sr, freq_output_plot):
    if freq_output_plot:
        fig, ax = plt.subplots()
        ax.plot(freq, 10 * np.log10(abs(Sl[i])), label='Left ear signal')
        ax.plot(freq, 10 * np.log10(abs(Sr[i])), label='Right ear signal')
        ax.set_xscale('log')
        plt.legend()
        # ax.set_ylim(-100, -50)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (dB)')
        plt.title('Frequency domain signal obtained after spherical convolution')
        plt.show()

def plot_grid(grid, grid_plot):
    if grid_plot:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")

        # Creating plot
        ax.scatter3D(grid[0], grid[1], grid[2], color="green")
        # ax.set_xlabel('')
        # norm  = grid[0]**2+(grid[1]+0.9)**2+grid[2]**2

