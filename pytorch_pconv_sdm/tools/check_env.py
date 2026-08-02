import platform

import torch


def main():
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
