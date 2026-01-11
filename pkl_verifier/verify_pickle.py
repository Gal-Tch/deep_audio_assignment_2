import pickle as pkl
import soundfile as sf
import os

def verify_data():
    pkl_path = 'force_align.pkl'
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        data = pkl.load(f)

    # 1. Print all keys and the specific ground truth text
    print("\n--- Pickle File Contents ---")
    for key in data.keys():
        if key == 'acoustic_model_out_probs':
            print(f"Key: '{key}' | Shape: {data[key].shape}")
        elif key == 'Audio':
            print(f"Key: '{key}' | Length: {len(data[key])} samples")
        else:
            print(f"Key: '{key}' | Value: {data[key]}")

    # 2. Export the audio to a WAV file so you can listen to it
    audio_key = 'audio' if 'audio' in data else 'Audio'
    audio_data = data[audio_key]
    sample_rate = 16000 # As specified in the assignment
    output_wav = 'results/verify_audio.wav'
    
    if not os.path.exists('results'):
        os.makedirs('results')
        
    sf.write(output_wav, audio_data, sample_rate)
    print(f"\n--- Audio Exported ---")
    print(f"Audio has been saved to: {output_wav}")
    print("You can now listen to this file to verify the ground truth text.")

if __name__ == "__main__":
    verify_data()
