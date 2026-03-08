import os
import platform
import stat
import urllib.request
import zipfile


def download_protoc() -> None:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        arch = "win64" if machine in ["amd64", "x86_64"] else "win32"
    elif system == "darwin":
        arch = "osx-universal_binary" if machine == "arm64" else "osx-x86_64"
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

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    protoc_dir = os.path.join(repo_root, "protoc")
    os.makedirs(protoc_dir, exist_ok=True)

    download_path = os.path.join(protoc_dir, filename)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, download_path)

    print("Extracting...")
    with zipfile.ZipFile(download_path, "r") as zf:
        zf.extractall(protoc_dir)

    bin_name = "protoc.exe" if system == "windows" else "protoc"
    bin_path = os.path.join(protoc_dir, "bin", bin_name)
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Expected protoc binary at: {bin_path}")

    if system != "windows":
        current_mode = os.stat(bin_path).st_mode
        os.chmod(bin_path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    os.remove(download_path)
    print(f"protoc installed to {bin_path}")


if __name__ == "__main__":
    download_protoc()
