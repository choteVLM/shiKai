from setuptools import setup, find_packages

setup(
    name="smolVLM",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pillow",
        "numpy",
        "opencv-python",
        "qdrant-client",
        "pyyaml",
    ],
    author="SmolVLM Team",
    author_email="",
    description="SmolVLM video inference package",
    keywords="video, vision, language, model",
    python_requires=">=3.7",
) 