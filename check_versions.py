import subprocess
import pkg_resources
import torch
import sys

def get_installed_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "Not installed"

def check_and_save_versions():
    # List of packages used in your DeepFake detection project
    packages = [
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'timm',
        'Pillow',
        'tqdm',
        'scikit-learn',
        'albumentations'
    ]

    # Get system info
    system_info = [
        f"Python version: {sys.version.split()[0]}",
        f"CUDA available: {torch.cuda.is_available()}"
    ]
    
    if torch.cuda.is_available():
        system_info.extend([
            f"CUDA version: {torch.version.cuda}",
            f"GPU: {torch.cuda.get_device_name(0)}"
        ])

    # Get package versions
    package_versions = []
    for package in packages:
        version = get_installed_version(package)
        package_versions.append(f"{package}=={version}")

    # Save to requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(package_versions))

    # Save detailed info to versions_info.txt
    with open('versions_info.txt', 'w') as f:
        f.write("System Information:\n")
        f.write('-' * 50 + '\n')
        f.write('\n'.join(system_info))
        f.write('\n\n')
        
        f.write("Package Versions:\n")
        f.write('-' * 50 + '\n')
        f.write('\n'.join(package_versions))

if __name__ == "__main__":
    print("Checking installed versions...")
    check_and_save_versions()
    print("Versions have been saved to 'requirements.txt' and 'versions_info.txt'") 