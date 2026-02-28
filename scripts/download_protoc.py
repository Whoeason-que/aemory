import urllib.request
import zipfile
import tarfile
import platform
import os
import sys

def download_protoc():
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        arch = "win64" if machine in ["amd64", "x86_64"] else "win32"
        ext = "zip"
    elif system == "darwin":
        arch = "osx-universal_binary" if machine == "arm64" else "osx-x86_64"
        ext = "zip"
    else:
        if machine == "aarch64":
            arch = "linux-aarch_64"
        elif machine == "armv7l":
            arch = "linux-armv7"
        else:
            arch = "linux-x86_64"
        ext = "zip"
    
    version = "25.5"
    filename = f"protoc-{version}-{arch}.{ext}"
    url = f"https://github.com/protocolbuffers/protobuf/releases/download/v{version}/{filename}"
    
    protoc_dir = os.path.join(os.path.dirname(__file__), "protoc")
    os.makedirs(protoc_dir, exist_ok=True)
    
    download_path = os.path.join(protoc_dir, filename)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, download_path)
    
    print(f"Extracting...")
    if ext == "zip":
        with zipfile.ZipFile(download_path, 'r') as zf:
            zf.extractall(protoc_dir)
    else:
        with tarfile.open(download_path, 'r:gz') as tf:
            tf.extractall(protoc_dir)
    
    bin_name = "protoc.exe" if system == "windows" else "protoc"
    src = os.path.join(protoc_dir, "bin", bin_name)
    dst = os.path.join(protoc_dir, "protoc.exe" if system == "windows" else "protoc")
    if os.path.exists(src):
        os.rename(src, dst)
    
    os.remove(download_path)
    print(f"protoc installed to {dst}")

if __name__ == "__main__":
    download_protoc()
