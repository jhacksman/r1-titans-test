from setuptools import setup, find_packages

setup(
    name="r1-titans-memory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "annoy",  # for approximate nearest neighbor search
    ],
    python_requires=">=3.8",
    author="Devin",
    description="Titans-inspired memory modules for DeepSeek R1",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
