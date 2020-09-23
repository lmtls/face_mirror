# FaceMirror
An OpenCV driven python program, which finds human/cat faces and mirrors them.

## How it works
![Initial image](https://i.imgur.com/upMVs35.jpg)
![Left mirrored image](https://i.imgur.com/0aJYFp4.jpg)
![Right mirrored image](https://i.imgur.com/4cUfi9h.jpg)


## Requirements
opencv-python 4.3.0.36

Pillow==7.2.0

numpy 1.19.1


## Setup
```
git clone https://github.com/lmtls/face_mirror
pip3 install -r requirements.txt
```


## Use
'python3 "face side choice" "path to initial image" "open-cv cascade file" "image scale parameter"'

Example: 'python3 -r initial.jpg haarcascade_frontalface_default.xml 1.1'


## Discord Bot
Program has a Discord bot version called "ShronkBot"
[ShronkBot official github repository](https://github.com/lmtls/ShronkBot)
