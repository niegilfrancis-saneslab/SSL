import pandas as pd
from scipy.io import wavfile
import glob
import re
import tqdm
import numpy as np
from scipy.signal import butter, filtfilt, welch, spectrogram
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec


experiments = [385]
headmic_channels = [118,35]

# Functions to sort file names correctly
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def bandpass(signal, sr, low=20000, high=50000, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)


# Define frequency bands (Hz) - to check highest power bands
width = 2500
f_min = 5000
f_max = 50000
bands = [(i, i + width) for i in range(f_min, f_max, width)]


def high_power_range(audio, sr, top_n = 4):
    # Power Spectral Density
    freqs, psd = welch(audio, sr, nperseg=512)


    band_powers = {}

    for low, high in bands:
        idx = np.logical_and(freqs >= low, freqs < high)
        band_powers[f"{low}-{high}"] = np.sum(psd[idx])

    # Sort bands by power
    sorted_bands = sorted(band_powers.items(), key=lambda x: x[1], reverse=True)

    min_f = 100000
    max_f = 0 
    for i in np.array(sorted_bands)[:top_n,0]:
        if int(i.split("-")[0]) < min_f:
            min_f = int(i.split("-")[0])
        if int(i.split("-")[1]) > max_f:
            max_f = int(i.split("-")[1])

    # getting the average of the bottom _n band power
    bot_n_avg_psd = np.mean(np.float64(np.array(sorted_bands)[top_n:,1]))
    # getting the std of the bottom _n band power
    bot_n_std_psd = np.std(np.float64(np.array(sorted_bands)[top_n:,1]))

    return min_f,max_f,bot_n_avg_psd,bot_n_std_psd



# Taking into consideration background noise
def signal_with_highest_peak_threshold_power(sig1, sig2, threshold_ratio=0.0002):

    def peak_region_score(signal):

        abs_sig = np.abs(signal)

        # estimate noise floor
        noise = np.median(abs_sig)

        peak_idx = np.argmax(abs_sig)
        peak_val = abs_sig[peak_idx]

        threshold = max(threshold_ratio * peak_val, noise * 3)

        # expand left
        left = peak_idx
        while left > 0 and abs_sig[left] >= threshold:
            left -= 1

        # expand right
        right = peak_idx
        while right < len(signal)-1 and abs_sig[right] >= threshold:
            right += 1

        start = left + 1
        end = right

        region = signal[start:end]

        signal_power = np.mean(region**2)
        noise_power = np.mean(signal[:start]**2)

        snr = signal_power / (noise_power + 1e-12)

        return snr, peak_idx, start, end


    s1, peak1, start1, end1 = peak_region_score(sig1)
    s2, peak2, start2, end2 = peak_region_score(sig2)

    winner = 1 if s1 > s2 else 2

    return winner, {
        "signal_1": {
            "score": s1,
            "peak_index": peak1,
            "cutoff_start": start1,
            "cutoff_end": end1,
        },
        "signal_2": {
            "score": s2,
            "peak_index": peak2,
            "cutoff_start": start2,
            "cutoff_end": end2,
        },
    }



def spectrogram_review_gui(sig1, sig2, fs):
    """
    Display original signals and spectrograms for two ultrasonic signals.
    Winner is determined based on vocalization energy.
    Allows user to switch the winner manually.
    """

    decision = {"value": None}

    # ----------------------------
    # Compute vocalization energy
    # ----------------------------
    def vocalization_energy(signal):
        f, t, Sxx = spectrogram(signal, fs, nperseg=512, noverlap=256, nfft=1024)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        med = np.median(Sxx_db, axis=1, keepdims=True)
        mad = np.median(np.abs(Sxx_db - med), axis=1, keepdims=True)
        mask = Sxx_db > med + 3 * mad
        energy = np.sum(Sxx[mask])
        return energy, f, t, Sxx

    # Compute energies
    energy1, f1, t1, Sxx1 = vocalization_energy(sig1)
    energy2, f2, t2, Sxx2 = vocalization_energy(sig2)

    winner = [1 if energy1 > energy2 else 2]  # mutable

    # ----------------------------
    # Determine y-axis limits for waveforms
    # ----------------------------
    global_min = min(sig1.min(), sig2.min())
    global_max = max(sig1.max(), sig2.max())

    # ----------------------------
    # Plot signals and spectrograms
    # ----------------------------
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), gridspec_kw={'height_ratios':[1,1,1,1]}, sharex=False)

    # Time-domain waveforms
    times1 = np.arange(len(sig1)) / fs
    axes[0].plot(times1, sig1, color='black')
    axes[0].set_title("Signal 1 | Time Domain")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_ylim(global_min, global_max)  # same scale

    times2 = np.arange(len(sig2)) / fs
    axes[1].plot(times2, sig2, color='black')
    axes[1].set_title("Signal 2 | Time Domain")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_ylim(global_min, global_max)  # same scale

    # Spectrograms
    def plot_spectrogram(ax, f, t, Sxx):
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        db_max = Sxx_db.max()
        db_min = db_max - 40
        Sxx_db = np.clip(Sxx_db, db_min, db_max)
        im = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="inferno")
        ax.set_yscale("log")
        ax.set_facecolor("black")
        ax.set_ylabel("Frequency (Hz)")
        return im

    im1 = plot_spectrogram(axes[2], f1, t1, Sxx1)
    axes[2].set_title("Signal 1 | Spectrogram")
    axes[2].set_xlabel("Time (s)")
    fig.colorbar(im1, ax=axes[2], orientation="vertical", label="Power (dB)")

    im2 = plot_spectrogram(axes[3], f2, t2, Sxx2)
    axes[3].set_title("Signal 2 | Spectrogram")
    axes[3].set_xlabel("Time (s)")
    fig.colorbar(im2, ax=axes[3], orientation="vertical", label="Power (dB)")

    # ----------------------------
    # Highlight winner
    # ----------------------------
    def update_highlight():
        if winner[0] == 1:
            axes[0].patch.set_edgecolor("lime")
            axes[0].patch.set_linewidth(2)
            axes[1].patch.set_edgecolor("none")
        else:
            axes[1].patch.set_edgecolor("lime")
            axes[1].patch.set_linewidth(2)
            axes[0].patch.set_edgecolor("none")
        fig.canvas.draw_idle()

    update_highlight()

    # ----------------------------
    # Buttons
    # ----------------------------
    ax_correct = plt.axes([0.15, 0.01, 0.2, 0.06])
    ax_wrong = plt.axes([0.55, 0.01, 0.2, 0.06])
    ax_switch = plt.axes([0.35, 0.01, 0.15, 0.06])

    btn_correct = Button(ax_correct, "Single Vox", color="lightgreen")
    btn_wrong = Button(ax_wrong, "Multiple Vox", color="lightcoral")
    btn_switch = Button(ax_switch, "Switch Winner", color="lightblue")

    def on_correct(event):
        decision["value"] = True
        plt.close(fig)

    def on_wrong(event):
        decision["value"] = False
        plt.close(fig)

    def on_switch(event):
        winner[0] = 2 if winner[0] == 1 else 1
        update_highlight()

    btn_correct.on_clicked(on_correct)
    btn_wrong.on_clicked(on_wrong)
    btn_switch.on_clicked(on_switch)

    plt.tight_layout()
    plt.show()

    return decision["value"] ,winner[0]

for exp in experiments:
    general_path = "D:/big_setup/experiment_{}/concatenated_data_cam_mic_sync/".format(exp)

    # getting the tracks
    track_files = glob.glob(general_path+"corrected_tracks/*.csv")
    track_files.sort(key=natural_keys)

    # getting the das files
    das_files = glob.glob(f"//sanesstorage.cns.nyu.edu/archive/Niegil/das/training_data/data/exp_{exp}*annotations.csv")
    das_files.sort(key=natural_keys)

    # getting the headmic files
    headmic_files = {}
    for idx in headmic_channels:
        temp_str = str(idx)
        headmic_files[idx] = glob.glob(general_path+f"headmic_{temp_str}_*.wav")
        headmic_files[idx].sort(key=natural_keys)


    length_of_experiment = len(glob.glob(general_path+"*.mp4"))
    powers = []
    powers_ = []

    for idx in tqdm.tqdm(range(length_of_experiment)):
        headmics = {}
        # getting the headmic channels
        for i in headmic_channels:
            file = headmic_files[i][idx]
            file_name = file.split("\\")[-1]
            if "%03d"%(idx) in file:
                sr,headmics[i] = wavfile.read(file)
            else:
                raise Exception(f"Error finding headmic file({file}) for idx {idx}")
        # getting the das file
        index = [id for id, s in enumerate(das_files) if "%03d"%(idx) in s]
        if len(index) == 1:
            file = das_files[index[0]]
            vox_times = pd.read_csv(file) 
            vox_times["label"] = None
        # pass
        else:
            print(f"Number of DAS files found for index {idx} is {len(index)}")
            raise Exception(f"More than one or less than one of das annotations files found for index {idx}")
        all_vox = []
        try:
            for vox_idx,start,stop in zip(vox_times[vox_times["name"]=="vox"].index,vox_times[vox_times["name"]=="vox"]["start_seconds"],vox_times[vox_times["name"]=="vox"]["stop_seconds"]):

                # This part is to take care of delays/ sync issues where the external mics and the audio is not aligned
                diff=stop-start
                if diff < 0.06:
                    add_amt = 0.06 - diff
                    stop = stop+add_amt/2
                    start = start-add_amt/2
                
                start_idx = int(sr*start)
                stop_idx = int(sr*stop)
                audio_0 = headmics[headmic_channels[0]][start_idx:stop_idx]
                audio_1 = headmics[headmic_channels[1]][start_idx:stop_idx]
                
                if (np.min(audio_0) == 0 and np.max(audio_0==0)) or (np.min(audio_1) == 0 and np.max(audio_1==0)):
                    break

                
                min_f_0,max_f_0,noise_pwr_0,noise_std_0 = high_power_range(audio_0,sr)
                min_f_1,max_f_1,noise_pwr_1,noise_std_1 = high_power_range(audio_1,sr)

                
                # audio_0 = bandpass(audio_0,sr,min_f_0,max_f_0)
                # audio_1 = bandpass(audio_1,sr,min_f_1,max_f_1)
                
                #winner, _  = signal_with_highest_band_power(audio_0,audio_1,sr,[min(min_f_0,min_f_1),max(max_f_0,max_f_1)],512)
                # winner, info = signal_with_highest_peak_threshold_power(audio_0,audio_1)

                user_ok, winner = spectrogram_review_gui(audio_0, audio_1, sr)

                if user_ok:
                    print("Single Vocalization ✅")
                    if winner == 1:
                        vox_times.loc[vox_idx,"label"] = 118
                    elif winner == 2:
                        vox_times.loc[vox_idx,"label"] = 35
                else:
                    print("Multiple Vocalizations ❌")

        except KeyboardInterrupt:
            vox_times.to_csv(file.split(".csv")[0]+"_gt.csv")

        vox_times.to_csv(file.split(".csv")[0]+"_gt.csv")   



            


        """
            all_vox.append(audio_0[:info["signal_1"]["cutoff_start"]])
            all_vox.append(np.zeros(100))
            all_vox.append(audio_0[info["signal_1"]["cutoff_start"]:info["signal_1"]["cutoff_end"]])
            all_vox.append(np.zeros(100))
            all_vox.append(audio_0[info["signal_1"]["cutoff_end"]:])
            if winner == 1:
                all_vox.append(np.ones(1)*10)
            all_vox.append(np.zeros(1000))



            # all_vox.append(audio_1)
            all_vox.append(audio_1[:info["signal_2"]["cutoff_start"]])
            all_vox.append(np.zeros(100))
            all_vox.append(audio_1[info["signal_2"]["cutoff_start"]:info["signal_2"]["cutoff_end"]])
            all_vox.append(np.zeros(100))
            all_vox.append(audio_1[info["signal_2"]["cutoff_end"]:])
            if winner == 2:
                all_vox.append(np.ones(1)*10)
            all_vox.append(np.zeros(1000))
        try:
            wavfile.write(general_path+f"{idx}_test.wav",rate=sr, data=np.concatenate(all_vox).ravel())
            print("Written")
        except:
            continue"""
