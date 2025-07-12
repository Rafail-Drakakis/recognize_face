# Face Recognition CLI

This repository provides simple command line utilities for detecting and recognizing faces in images using the [face_recognition](https://github.com/ageitgey/face_recognition) library.

## Features

- **Detect** faces in a single image using either the HOG or CNN model.
- **Recognize** a known face within another image.
- Draw bounding boxes around detected or recognized faces and optionally save or display the result.

Sample images (`toby.jpg` and `office.jpg`) are included for experimentation.

## Installation

1. Ensure Python 3.8+ is installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The `face_recognition` package requires `cmake` and dlib to build. On Debian/Ubuntu systems you can install the build tools with:

```bash
apt-get update && apt-get install -y build-essential cmake
```

Consult the [face_recognition documentation](https://github.com/ageitgey/face_recognition#installation) if you encounter issues building dlib.

## Usage

```
python recognize_face.py detect <image> [--model hog|cnn] [--output OUTPUT] [--no-show]
python recognize_face.py recognize --known <known_image> --unknown <image> [--model hog|cnn] [--output OUTPUT] [--no-show]
```

- `--model`: choose between the faster `hog` model or the more accurate but slower `cnn` model (requires a GPU for reasonable performance).
- `--output`: save the annotated image to a file instead of or in addition to displaying it.
- `--no-show`: suppress opening a window to show the result.

### Examples

Detect faces in `office.jpg` using the default HOG model:

```bash
python recognize_face.py detect office.jpg
```

Recognize `toby.jpg` within `office.jpg` and save the result:

```bash
python recognize_face.py recognize --known toby.jpg --unknown office.jpg --output result.jpg
```

## Notes

- The included sample images are courtesy of *The Office* television show and are for demonstration purposes only.
