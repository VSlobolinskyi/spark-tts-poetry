import os
import time
import subprocess
import shutil
from pathlib import Path

def setup_triton_server(pretrained_model_dir="pretrained_models/Spark-TTS-0.5B"):
    """Set up Triton server model repository and start the server."""
    # Define paths
    src_model_repo = Path("runtime/triton_trtllm/model_repo")
    dest_model_repo = Path("triton_model_repo")
    
    # Create model repository
    os.makedirs(dest_model_repo, exist_ok=True)
    
    # Copy model repository structure from template
    if src_model_repo.exists():
        print(f"Copying model repository from {src_model_repo} to {dest_model_repo}")
        # Copy each component
        for component in ["audio_tokenizer", "spark_tts", "vocoder", "tensorrt_llm"]:
            src_component = src_model_repo / component
            dest_component = dest_model_repo / component
            
            # Create component directories
            os.makedirs(dest_component / "1", exist_ok=True)
            
            # Copy model.py if it exists
            if (src_component / "1" / "model.py").exists():
                shutil.copy(
                    src_component / "1" / "model.py",
                    dest_component / "1" / "model.py"
                )
            
            # Copy and configure config.pbtxt
            if (src_component / "config.pbtxt").exists():
                # Read template
                with open(src_component / "config.pbtxt", "r") as f:
                    config_content = f.read()
                
                # Replace template variables
                config_content = config_content.replace("${model_dir}", str(Path(pretrained_model_dir).absolute()))
                config_content = config_content.replace("${llm_tokenizer_dir}", str(Path(pretrained_model_dir) / "LLM"))
                config_content = config_content.replace("${triton_max_batch_size}", "16")
                config_content = config_content.replace("${decoupled_mode}", "False")  # Default to non-streaming
                config_content = config_content.replace("${max_queue_delay_microseconds}", "0")
                config_content = config_content.replace("${audio_chunk_duration}", "1.0")
                config_content = config_content.replace("${max_audio_chunk_duration}", "30.0")
                config_content = config_content.replace("${audio_chunk_size_scale_factor}", "8.0")
                config_content = config_content.replace("${audio_chunk_overlap_duration}", "0.1")
                config_content = config_content.replace("${audio_tokenizer_frame_rate}", "75")
                
                # Write updated config
                with open(dest_component / "config.pbtxt", "w") as f:
                    f.write(config_content)
    else:
        raise FileNotFoundError(f"Model repository template not found at {src_model_repo}")
    
    # Start Triton server
    print("Starting Triton server...")
    cmd = ["tritonserver", "--model-repository", str(dest_model_repo)]
    
    # Use Popen to start server in background
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for server to initialize
    time.sleep(5)
    
    # Check if server is running
    if process.poll() is not None:
        # Server terminated
        stdout, stderr = process.communicate()
        raise RuntimeError(f"Triton server failed to start: {stderr.decode() if stderr else 'Unknown error'}")
    
    print("Triton server started successfully")
    return process, "localhost:8000"  # Return process and HTTP endpoint