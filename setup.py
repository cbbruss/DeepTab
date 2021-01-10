import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frustanet", # Replace with your own username
    version="0.0.1",
    author="Bayan Bruss",
    author_email="cbbruss@example.com",
    description="A Neural Network for Tabular Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cbbruss/frustanet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)