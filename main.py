# main.py

import os
import sys
import subprocess
import glob
import argparse
from audio_processing import process_audio

# Define paths to the cloned repositories
SADTALKER_PATH = os.path.join(os.getcwd(), 'SadTalker')
WAV2LIP_PATH = os.path.join(os.getcwd(), 'Wav2Lip')

def run_command(command, cwd=None):
    """Executes a shell command and raises an error if it fails."""
    try:
        print(f"Executing: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}\n{e}")
        sys.exit(1)

def main(args):
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # === Step 1: (Optional) Audio Pre-processing ===
    driven_audio = args.driven_audio
    if not args.skip_audio_processing:
        enhanced_audio_path = os.path.join(args.output_dir, os.path.basename(driven_audio).replace(".wav", "_enhanced.wav"))
        driven_audio = process_audio(driven_audio, enhanced_audio_path, visualize=args.visualize_audio)
    else:
        print("\n--- Skipping audio pre-processing. ---")
    
    # === Step 2: SadTalker Inference ===
    print("\n--- Step 1/3: Running SadTalker Inference ---")
    sadtalker_output_dir = os.path.join(args.output_dir, 'sadtalker_output')
    sadtalker_cmd = [
        'python', 'inference.py',
        '--driven_audio', driven_audio,
        '--source_image', args.source_image,
        '--result_dir', sadtalker_output_dir,
        '--enhancer', 'gfpgan'
    ]
    run_command(sadtalker_cmd, cwd=SADTALKER_PATH)
    
    # Find the generated video file
    sadtalker_videos = glob.glob(os.path.join(sadtalker_output_dir, '*.mp4'))
    if not sadtalker_videos:
        print("Error: SadTalker did not produce a video file.")
        sys.exit(1)
    sadtalker_video = sorted(sadtalker_videos, key=os.path.getmtime, reverse=True)[0]
    print(f"  - SadTalker video generated: {sadtalker_video}")
    
    # === Step 3: Wav2Lip Enhancement ===
    print("\n--- Step 2/3: Running Wav2Lip Enhancement ---")
    wav2lip_output_video = os.path.join(args.output_dir, 'sadtalker_wav2lip.mp4')
    # Note: Use the ORIGINAL audio for Wav2Lip for best sync results on its model
    wav2lip_cmd = [
        'python', 'inference.py',
        '--checkpoint_path', os.path.join(WAV2LIP_PATH, 'checkpoints/Wav2Lip-SD-GAN.pt'),
        '--face', sadtalker_video,
        '--audio', args.driven_audio,
        '--outfile', wav2lip_output_video
    ]
    run_command(wav2lip_cmd, cwd=WAV2LIP_PATH)
    print(f"  - Wav2Lip enhanced video saved to: {wav2lip_output_video}")

    # === Step 4: (Optional) Post-processing ===
    final_video = wav2lip_output_video
    if not args.skip_post_processing:
        print("\n--- Step 3/3: Running Post-processing ---")
        post_processed_video = os.path.join(args.output_dir, 'final_enhanced_video.mp4')
        
        # Using ffmpeg for frame interpolation and super-resolution
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', wav2lip_output_video,
            '-filter:v', 'minterpolate=fps=30,scale=512:512:flags=lanczos',
            post_processed_video
        ]
        run_command(ffmpeg_cmd)
        final_video = post_processed_video
        print(f"  - Post-processing complete. Final video saved to: {final_video}")
    else:
        print("\n--- Skipping post-processing. ---")
        final_video = os.path.join(args.output_dir, 'final_video.mp4')
        os.rename(wav2lip_output_video, final_video)

    print(f"\n✅ Pipeline finished! Final video is available at: {final_video}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="High-Fidelity Speech-to-Face Pipeline")
    parser.add_argument('--source_image', type=str, required=True, help='Path to the source face image.')
    parser.add_argument('--driven_audio', type=str, required=True, help='Path to the driving audio file (.wav).')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the results.')

    parser.add_argument('--skip_audio_processing', action='store_true', help='Skip the audio pre-processing step.')
    parser.add_argument('--visualize_audio', action='store_true', help='Generate a visualization of the audio processing.')
    parser.add_argument('--skip_post_processing', action='store_true', help='Skip the final post-processing (frame interpolation and super-resolution).')
    
    args = parser.parse_args()
    
    # Check for submodule paths
    if not os.path.exists(SADTALKER_PATH):
        print(f"Error: SadTalker repository not found at '{SADTALKER_PATH}'")
        print("Please clone it using: git clone https://github.com/OpenTalker/SadTalker.git")
        sys.exit(1)
        
    if not os.path.exists(WAV2LIP_PATH):
        print(f"Error: Wav2Lip repository not found at '{WAV2LIP_PATH}'")
        print("Please clone it using: git clone https://github.com/Rudrabha/Wav2Lip.git")
        sys.exit(1)
        
    main(args)