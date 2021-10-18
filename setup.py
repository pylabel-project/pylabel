from setuptools import find_packages, setup

setup(
    name='pylabelalphatest',
    packages=['pylabelalpha'],
    version='0.1.5',
    description='My first Python library',
    author='Me',
    license='MIT',
    install_requires=['pandas','bbox_visualizer'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)