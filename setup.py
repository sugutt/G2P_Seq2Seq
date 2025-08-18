from setuptools import setup, find_packages
import os

# Create required folders if they don't exist
def ensure_dirs():
    folders = [
        'data',
        'models',
        'outputs',
        'notebooks',
        'src'
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")

# Read requirements.txt if it exists
requirements = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

ensure_dirs()

setup(
    name='G2P_Seq2Seq',
    version='0.1',
    description='Grapheme-to-Phoneme Seq2Seq Project',
    author='',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=requirements,
    include_package_data=True,
    python_requires='>=3.7',
)
