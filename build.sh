rm -r build/
rm -r pylabel.egg-info/
rm -r dist/
Python setup.py sdist bdist_wheel 
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
