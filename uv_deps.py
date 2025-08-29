import toml
import subprocess
import sys

def install_uv_packages():
    config = toml.load("pyproject.toml")
    uv_packages = config.get("tool", {}).get("spark-tts", {}).get("uv-packages", {})
    
    for package, url in uv_packages.items():
        print(f"Installing {package} with uv...")
        subprocess.run([sys.executable, "-m", "uv", "pip", "install", url], check=True)

if __name__ == "__main__":
    install_uv_packages()