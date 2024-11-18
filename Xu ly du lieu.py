import wfdb
import pandas as pd
import numpy as np
from scipy import signal


def load_mitbih_record(record_name, path=''):
    """
    Load MIT-BIH record using wfdb

    Parameters:
    record_name : str
        Name of the record to load
    path : str
        Path to the record files

    Returns:
    signals : ndarray
        ECG signals
    fields : dict
        Signal information
    annotations : ndarray
        Beat annotations
    """
    # Read the signal
    record = wfdb.rdrecord(path + record_name)
    # Read annotations
    annotations = wfdb.rdann(path + record_name, 'atr')

    return record.p_signal, record.__dict__, annotations


def process_mitbih_data(record_name, path=''):
    """
    Process MIT-BIH data and convert to DataFrame

    Parameters:
    record_name : str
        Name of the record to process
    path : str
        Path to the record files

    Returns:
    df : pandas.DataFrame
        Processed data in DataFrame format
    """
    # Load the data
    signals, fields, annotations = load_mitbih_record(record_name, path)

    # Create time array (sampling frequency is typically 360 Hz)
    time = np.arange(len(signals)) / fields['fs']

    # Create DataFrame
    df = pd.DataFrame({
        'Time': time,
        'ECG_I': signals[:, 0],  # First channel (usually MLII)
        'ECG_II': signals[:, 1]  # Second channel (usually V1)
    })

    # Add annotations
    ann_time = annotations.sample / fields['fs']
    ann_type = annotations.symbol

    # Create annotations DataFrame
    df_ann = pd.DataFrame({
        'Time': ann_time,
        'Annotation': ann_type
    })

    return df, df_ann


def save_to_excel(record_name, output_path, chunk_size=None):
    """
    Save MIT-BIH data to Excel file(s)

    Parameters:
    record_name : str
        Name of the record to process
    output_path : str
        Path to save Excel files
    chunk_size : int, optional
        Number of samples per Excel file. If None, save as single file
    """
    # Process the data
    df_signal, df_ann = process_mitbih_data(record_name)

    if chunk_size is None:
        # Save as single file
        with pd.ExcelWriter(f'{output_path}/{record_name}_data.xlsx') as writer:
            df_signal.to_excel(writer, sheet_name='ECG_Signals', index=False)
            df_ann.to_excel(writer, sheet_name='Annotations', index=False)
    else:
        # Save in chunks
        num_chunks = len(df_signal) // chunk_size + 1
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df_signal))

            # Get data chunk
            chunk_signal = df_signal.iloc[start_idx:end_idx]

            # Get corresponding annotations
            mask = (df_ann['Time'] >= chunk_signal['Time'].iloc[0]) & \
                   (df_ann['Time'] <= chunk_signal['Time'].iloc[-1])
            chunk_ann = df_ann[mask]

            # Save chunk
            with pd.ExcelWriter(f'{output_path}/{record_name}_chunk_{i + 1}.xlsx') as writer:
                chunk_signal.to_excel(writer, sheet_name='ECG_Signals', index=False)
                chunk_ann.to_excel(writer, sheet_name='Annotations', index=False)


def main():
    # Example usage
    record_name = '118e00'  # MIT-BIH record number
    path = 'C:\\Users\\ACER\\PycharmProjects\\DSP\\ECGdataset\\mit-bih-noise-stress-test-database-1.0.0'  # Path to MIT-BIH database
    output_path = 'C:\\Users\\ACER\\PycharmProjects\\DSP\\ECGdataset\\output-excel'  # Path to save Excel files

    # Save as single file
    save_to_excel(record_name, output_path)

    # Or save in chunks of 100,000 samples each
    save_to_excel(record_name, output_path, chunk_size=100000)


if __name__ == '__main__':
    main()