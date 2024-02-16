import face_recognition as fr
import cv2

# for loading images from the drive
# image1 = fr.load_image_file(r"C:\Users\DELL\OneDrive\Desktop\Ml-imges\rohit1.JPG")
image1 = fr.load_image_file(r"C:\Users\satya\PycharmProjects\FACE-Recognition\pic\1.png")
image2 = fr.load_image_file(r"C:\Users\satya\PycharmProjects\FACE-Recognition\pic\4.jpeg")

# for changing the color of the image
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# encoding the face of the image
enc1 = fr.face_encodings(image1)[0]

face_locations = fr.face_locations(image2)
face_encoding = fr.face_encodings(image2,face_locations)

# Compare the single person face with each face in the group photo
matches = fr.compare_faces(face_encoding, enc1)
# Find the indices of the matched faces
matched_indices = [index for index, match in enumerate(matches) if match]

# Draw rectangles around the matched faces in the group photo
for index in matched_indices:
    top, right, bottom, left = face_locations[index]
    cv2.rectangle(image2, (left, top), (right, bottom), (0, 255, 0), 2)

# Display the group photo with matched faces (if any)
cv2.imshow("Matched Faces", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()