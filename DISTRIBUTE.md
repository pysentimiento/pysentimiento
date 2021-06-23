## Build instructions

0. Change version at `setup.py`
1. Build it

```
python -m build
```
2. Upload

```
python -m twine upload --repository pypi dist/*
```
