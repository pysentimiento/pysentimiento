import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysentimiento", # Replace with your own username
    version="0.0.1.2",
    author="Juan Manuel Pérez, Juan Carlos Giudici, Franco Luque",
    author_email="jmperez@dc.uba.ar",
    description="A Transformer-based library for Sentiment Analysis in Spanish",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/finiteautomata/pysentimiento",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "transformers>=3.5.1",
    ]
)
