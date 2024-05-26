# Audio-to-text
A simple python tool, using whisper to transcript.
## How to use
### Install
```
pip install -r requirements.txt
```
If you want to use cuda, install the pytorch with cuda additionally.\
If you want to use different model, modify the 
```
WHISPER_MODEL = 'base'
```
parameter to any model name that is available in the **Available models and languages** section in [this page](https://pypi.org/project/openai-whisper/).
### Run
```
python audio2text.py
```
