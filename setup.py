# Test
from setuptools import setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pylabel",
    packages=["pylabel"],
    version="0.1.40",
    description="Transform, analyze, and visualize computer vision annotations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pylabel-project/pylabel",
    author="PyLabel Project",
    license="MIT",
    install_requires=[
        "pandas",
        "bbox_visualizer",
        "matplotlib",
        "opencv-python",
        "scikit-learn",
        "jupyter_bbox_widget",
        "pyyaml",
    ],
)
