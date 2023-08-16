import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from sklearn.preprocessing import MinMaxScaler

def plot_prediction(x_pred, 
                    y_pred, 
                    y_true, 
                    normalise=False, 
                    add_peaks=False,
                    save=False):

    if normalise:
        #standardize y pred and true 
        y_pred = MinMaxScaler().fit_transform(y_pred.reshape(-1, 1)).squeeze()
        y_true = MinMaxScaler().fit_transform(y_true.reshape(-1, 1)).squeeze()

    fig, ax = plt.subplots()
    ax.plot(x_pred, y_pred, color='r', label='prediction')
    ax.plot(x_pred, y_true, color='black', label='TDDFT')

    if add_peaks:
        peaks, _ = find_peaks(y_pred)
        widths_half = peak_widths(y_pred, peaks, rel_height=0.1)
        print(x_pred[peaks], widths_half[0])
        for x_peak, y_peak, width in zip(x_pred[peaks], \
                       y_pred[peaks], widths_half[0]):
            ax.text(x_peak, y_peak + 1, f"width = {width:.2f}", size=15)


    ax.legend()
    ax.set_yticklabels([])
    ax.set_xlabel('Energies (eV)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.tick_params(axis='x', which='major', direction='out', 
                bottom=True, width=2, length=5)

    if save:
        plt.savefig('./spectra.png', dpi=300, bbox_inches='tight')