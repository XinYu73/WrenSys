import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wren_sys",
    version="0.0.1",
    author="Yu Xin",
    author_email="xy741496743@gmail.com",
    description="synthesizability evaluation based on Wren model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XinYu73/Wren_Sys.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)