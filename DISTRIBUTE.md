## Build instructions

0. Change version at `pyproject.toml`
1. Build it

```
poetry build
```
2. Upload

```
python -m twine upload --repository pypi dist/*
```
