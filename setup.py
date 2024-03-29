import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="karpathy-ai",
    version="0.1.0",
    author="Charlie Roth",
    author_email="charlieroth4@gmail.com",
    description="Repository for learning from Andrej Karpathy's AI videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/charlieroth/karpathy-ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)