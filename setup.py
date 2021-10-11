from setuptools import find_packages, setup

setup(
    name='pylabelalpha_test_1',
    packages=['pylabelalpha'],
    version='0.1.2',
    description='My first Python library',
    author='Me',
    license='MIT',
    install_requires=['pandas'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)