from setuptools import setup

setup(
    name='pylabel',
    packages=['pylabel'],
    version='0.1.4',
    description='Transform, analyze, and visualize computer vision annotations.',
    author='PyLabel Project',
    license='MIT',
    install_requires=['pandas','bbox_visualizer','matplotlib','opencv-python','scikit-learn']
)