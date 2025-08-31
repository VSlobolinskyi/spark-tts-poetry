import os
import torch
import soundfile as sf
import subprocess
from pathlib import Path

# Updated TritonSparkTTS class to work with the containerized setup
class TritonSparkTTS:
    """Wrapper for gRPC client that matches the SparkTTS API"""
    
    def __init__(self, server_url="localhost:8001", model_name="spark_tts"):
        self.server_url = server_url
        self.model_name = model_name
        self.sample_rate = 16000
    
    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path=None,
        prompt_text=None,
        gender=None,
        pitch=None,
        speed=None,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
    ):
        """Call client_grpc.py and return results as tensor"""
        if prompt_speech_path is None:
            raise ValueError("Reference audio (prompt_speech_path) is required")
        
        # Ensure output directory exists
        os.makedirs("./tmp", exist_ok=True)
        
        # Use your existing client_grpc.py - no changes needed since server ports are forwarded
        cmd = [
            "python", "runtime/triton_trtllm/client_grpc.py",
            "--server-addr", "localhost",
            "--reference-audio", str(prompt_speech_path),
            "--reference-text", prompt_text if prompt_text else "",
            "--target-text", text,
            "--model-name", self.model_name,
            "--log-dir", "./tmp"
        ]
        
        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Find the output audio file (should be test.wav)
        output_file = "./tmp/test.wav"
        if os.path.exists(output_file):
            # Load and return the audio
            wav, _ = sf.read(output_file)
            return torch.tensor(wav)
        else:
            # Error handling
            print(f"Error: Output file not found at {output_file}")
            print(f"Command output: {result.stdout}")
            print(f"Command error: {result.stderr}")
            return torch.tensor([0.0])