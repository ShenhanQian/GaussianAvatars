from setuptools import setup, find_packages

setup(
    name="mvht",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "ConfigArgParse",
        "torch",
        "torchvision",
        # "pytorch3d",
        "matplotlib==3.8.0",
        "tensorboard",
        "numpy==1.22.3",
        "scipy",
        "opencv-python",
        "chumpy",
        "face-alignment",
        "face-detection-tflite",
        "tyro",
        "pillow",
        'trimesh',
        'iopath',
        'pyyaml',
        'numpy==1.23.1',
    ],
)
