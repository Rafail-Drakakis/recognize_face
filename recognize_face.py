from PIL import Image, ImageDraw
import face_recognition
import numpy as np

def find_all_faces_in_image_hog(image_path):
    """
    The function `find_all_faces_in_image_hog` uses the Histogram of Oriented Gradients (HOG) algorithm
    to detect and display all faces in an image.
    
    :param image_path: The image_path parameter is the path to the image file that you want to find
    faces in. It should be a string representing the file path, including the file extension (e.g.,
    "path/to/image.jpg")
    """
    image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(image)

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location

        # Access the actual face:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()

def recognize_face_in_image(image_to_recognize, image_contained_face):
    """
    The `recognize_face_in_image` function takes in two images, one with a known face and one with
    unknown faces, and uses face recognition to identify and draw a box around any matching faces in the
    unknown image.
    
    :param image_to_recognize: The image file that contains the face you want to recognize. This image
    will be used to create a face encoding that will be compared to the faces in the other image
    :param image_contained_face: The parameter "image_contained_face" is the path or file object of the
    image that contains the face(s) you want to recognize
    """
    known_image = face_recognition.load_image_file(image_to_recognize)
    encoding = face_recognition.face_encodings(known_image)[0]

    # Load an image with unknown faces
    unknown_image = face_recognition.load_image_file(image_contained_face)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)

    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([encoding], face_encoding)

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance([encoding], face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:

            # Draw a box around the face using the Pillow module
            draw.rectangle(((left - 20, top - 20), (right + 20, bottom + 20)), outline=(0, 255, 0), width=20)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()

def find_all_faces_in_image_cnn(image_path):
    """
    The function `find_all_faces_in_image_cnn` uses a convolutional neural network (CNN) model to detect
    and display all faces in an image.
    
    :param image_path: The path to the image file that you want to find faces in
    """
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location

        # Access the actual face
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()

if __name__ == "__main__":
    choice = int(input("Enter\n1.To find all faces in an image with cnn algorithm\n2.To find all faces in an image with hog algorithm\n3.To see if a face is in an image: "))

    image_path = input("Enter the image filename: ")

    if choice == 1:
        find_all_faces_in_image_cnn(image_path)
    elif choice == 2:
        find_all_faces_in_image_hog(image_path)
    elif choice == 3:
        image_to_recognize = input("Enter the filename which contains the person's face: ")
        recognize_face_in_image(image_to_recognize, image_path)