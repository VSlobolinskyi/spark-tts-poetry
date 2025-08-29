import os
import sys
import time
import socket
from pathlib import Path
import subprocess
import soundfile as sf
import torch

try:
    import tritonclient.grpc as grpcclient
except Exception:
    grpcclient = None  # we'll still work, but readiness checks will be weaker

class TritonSparkTTS:
    """Wrapper for gRPC client that matches the SparkTTS API"""

    def __init__(self, server_url="localhost:8001", model_name="spark_tts", auto_start=False, project_root=None):
        # server_url form: "host:port"
        if ":" in server_url:
            host, port = server_url.split(":", 1)
            self.server_host = host
            self.server_port = int(port)
        else:
            self.server_host = server_url
            self.server_port = 8001
        self.server_url = f"{self.server_host}:{self.server_port}"
        self.model_name = model_name
        self.sample_rate = 16000
        self.auto_start = auto_start
        self.project_root = Path(project_root or Path.cwd())

    def _is_port_open(self, host, port, timeout=0.3):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            try:
                s.connect((host, port))
                return True
            except Exception:
                return False

    def _wait_for_triton(self, timeout=90):
        """Wait until Triton server is ready and model is ready."""
        # Quick TCP check first
        start = time.time()
        while time.time() - start < timeout:
            if self._is_port_open(self.server_host, self.server_port):
                break
            time.sleep(0.5)
        else:
            return False, "gRPC port not open"

        # If we have tritonclient, ask the server properly
        if grpcclient is not None:
            try:
                client = grpcclient.InferenceServerClient(url=self.server_url, verbose=False)
                # server ready
                for _ in range(int(timeout * 2)):  # ~0.5s steps
                    try:
                        if client.is_server_ready() and client.is_model_ready(self.model_name):
                            return True, ""
                    except Exception:
                        pass
                    time.sleep(0.5)
                return False, "model not ready"
            except Exception as e:
                return False, f"readiness probe failed: {e}"

        # Fallback: port open is all we can assert
        return True, ""

    def _maybe_start_server(self):
        """Start Triton using your script if auto_start is enabled."""
        if not self.auto_start:
            return
        # Start stages 0..3 offline, as we discussed earlier
        cmd = [
            "bash",
            str(self.project_root / "runtime" / "triton_trtllm" / "triton_run.sh"),
            "0", "3", "offline",
        ]
        subprocess.run(cmd, check=True)

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
        if not prompt_speech_path:
            raise ValueError("Reference audio (prompt_speech_path) is required")

        prompt_speech_path = str(prompt_speech_path)
        if not os.path.exists(prompt_speech_path):
            raise FileNotFoundError(f"Reference audio does not exist: {prompt_speech_path}")

        # Ensure server is up (start if requested)
        ok, why = self._wait_for_triton(timeout=10)
        if not ok:
            self._maybe_start_server()
            ok, why = self._wait_for_triton(timeout=90)
            if not ok:
                raise RuntimeError(f"Triton not ready at {self.server_url}: {why}")

        # Ensure output directory exists
        out_dir = self.project_root / "tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "test.wav"

        # Remove any stale file from previous runs
        try:
            if out_file.exists():
                out_file.unlink()
        except Exception:
            pass

        # Run client_grpc.py with explicit host/port/mode and this interpreter
        client = self.project_root / "runtime" / "triton_trtllm" / "client_grpc.py"
        cmd = [
            sys.executable, str(client),
            "--server-addr", self.server_host,
            "--server-port", str(self.server_port),
            "--reference-audio", prompt_speech_path,
            "--reference-text", prompt_text or "",
            "--target-text", text,
            "--model-name", self.model_name,
            "--num-tasks", "1",
            "--mode", "offline",
            "--log-dir", str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                "client_grpc.py failed\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

        if not out_file.exists():
            # Client ran but didn’t write the file (usually server-side error)
            raise FileNotFoundError(
                f"Expected output not found: {out_file}\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

        wav, sr = sf.read(str(out_file))
        if sr != self.sample_rate and sr > 0:
            # Keep it simple; most downstream code expects float32 mono
            wav = wav.astype("float32")
        return torch.tensor(wav, dtype=torch.float32)
