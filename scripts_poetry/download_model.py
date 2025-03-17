import os
import subprocess
import sys

def run_command(command, error_message):
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(error_message)
        sys.exit(1)

def main():
    # Create the pretrained_models directory if it doesn't exist
    os.makedirs("pretrained_models", exist_ok=True)
    
    # Install git-lfs (assumes git-lfs is installed on your system)
    print("Running 'git lfs install'...")
    run_command(["git", "lfs", "install"], "Error: Failed to run 'git lfs install'. Make sure git-lfs is installed (https://git-lfs.com).")
    
    # Clone the repository if it doesn't already exist
    clone_dir = os.path.join("pretrained_models", "Spark-TTS-0.5B")
    if not os.path.exists(clone_dir):
        print(f"Cloning repository into {clone_dir}...")
        run_command(
            ["git", "clone", "https://huggingface.co/SparkAudio/Spark-TTS-0.5B", clone_dir],
            "Error: Failed to clone the repository."
        )
    else:
        print(f"Directory '{clone_dir}' already exists. Skipping clone.")

if __name__ == "__main__":
    main()
