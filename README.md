# Run Tests

Run unit tests with `python -m pytest`

# Building

## Windows

```
pip install -r requirements.txt
pip install pyrtlsdrlib
```

Create portable .exe with:
```
pip install pyinstaller
python -m PyInstaller -F main.py
```
The .exe will be in /dist

## Ubuntu

```
sudo apt update
sudo apt install portaudio19-dev
pip install -r requirements.txt
```