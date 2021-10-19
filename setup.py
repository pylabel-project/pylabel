from setuptools import find_packages, setup

setup(
    name='pylabel',
    packages=['pylabel'],
    version='0.1.1',
    description='Transform, analyze, and visualize computer vision annotations.',
    author='PyLabel Project',
    license='MIT',
    install_requires=['pandas','bbox_visualizer'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)