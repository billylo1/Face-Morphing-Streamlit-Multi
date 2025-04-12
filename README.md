Face Morphing in Streamlit
===================

Adapts https://github.com/Azmarie/Face-Morphing into a Streamlit app.

## Installation on Ubuntu

- sudo apt install ffmpeg cmake libjpeg-dev zlib1g-dev
- python3 -m venv .
- source ./bin/activate
- pip3 install -r requirements.txt
- make sure Pillow is at 9.5 (error in 10.0+)

## Running

```
$ source ./bin/activate
$ streamlit run code/app.py
```

