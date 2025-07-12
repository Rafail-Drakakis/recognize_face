"""Face recognition command line utilities.

Provides two main operations:

- ``detect``: Detects faces in an image using either the HOG or CNN model and
  optionally saves or displays the result.
- ``recognize``: Given an image of a known face and an image that may contain
  that person, highlights matching faces.

Example::

    python recognize_face.py detect office.jpg --model hog
    python recognize_face.py recognize --known toby.jpg --unknown office.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw
import face_recognition
import numpy as np


FaceLocation = Tuple[int, int, int, int]


def load_image(path: Path) -> np.ndarray:
    """Load an image from *path* into a ``numpy`` array."""
    return face_recognition.load_image_file(str(path))


def detect_faces(image: np.ndarray, model: str = "hog") -> List[FaceLocation]:
    """Return face bounding boxes detected in *image*."""
    return face_recognition.face_locations(image, model=model)


def draw_boxes(
    image: np.ndarray,
    boxes: List[FaceLocation],
    color: Tuple[int, int, int] = (0, 255, 0),
    width: int = 4,
) -> Image.Image:
    """Return a ``PIL.Image`` with *boxes* drawn on ``image``."""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    for top, right, bottom, left in boxes:
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=width)
    del draw
    return pil_image


def recognize_faces(
    known_image_path: Path,
    unknown_image_path: Path,
    *,
    model: str = "hog",
) -> Tuple[np.ndarray, List[FaceLocation]]:
    """Return ``unknown_image`` and bounding boxes of faces matching ``known_image``."""
    known_image = load_image(known_image_path)
    known_encoding = face_recognition.face_encodings(known_image)[0]

    unknown_image = load_image(unknown_image_path)
    locations = detect_faces(unknown_image, model=model)
    encodings = face_recognition.face_encodings(unknown_image, locations)

    matches: List[FaceLocation] = []
    for location, encoding in zip(locations, encodings):
        is_match = face_recognition.compare_faces([known_encoding], encoding)[0]
        if is_match:
            matches.append(location)
    return unknown_image, matches


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face recognition utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    detect_p = sub.add_parser("detect", help="Detect faces in an image")
    detect_p.add_argument("image", type=Path, help="Image to analyze")
    detect_p.add_argument("--model", choices=["hog", "cnn"], default="hog")
    detect_p.add_argument("--output", type=Path, help="Save annotated image to file")
    detect_p.add_argument("--no-show", action="store_true", help="Do not open a window to display the result")

    rec_p = sub.add_parser("recognize", help="Recognize a known face in another image")
    rec_p.add_argument("--known", type=Path, required=True, help="Image containing the known face")
    rec_p.add_argument("--unknown", type=Path, required=True, help="Image that may contain the face")
    rec_p.add_argument("--model", choices=["hog", "cnn"], default="hog")
    rec_p.add_argument("--output", type=Path, help="Save annotated image to file")
    rec_p.add_argument("--no-show", action="store_true")

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "detect":
        image = load_image(args.image)
        boxes = detect_faces(image, model=args.model)
        print(f"Found {len(boxes)} face(s)")
        if args.output or not args.no_show:
            result = draw_boxes(image, boxes)
            if args.output:
                result.save(args.output)
            if not args.no_show:
                result.show()
    elif args.command == "recognize":
        image, matches = recognize_faces(args.known, args.unknown, model=args.model)
        print(f"Found {len(matches)} matching face(s)")
        if args.output or not args.no_show:
            result = draw_boxes(image, matches)
            if args.output:
                result.save(args.output)
            if not args.no_show:
                result.show()


if __name__ == "__main__":
    main()
