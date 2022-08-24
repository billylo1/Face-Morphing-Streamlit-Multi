import pathlib
import uuid

import cv2
import streamlit as st

from delaunay_triangulation import make_delaunay
from face_landmark_detection import generate_face_correspondences
from face_morph import generate_morph_sequence

IMAGES_DIR = pathlib.Path(__file__).parent.parent.joinpath("images/aligned_images").absolute()


def doMorphing(img1, img2, duration: float, frame_rate: int, draw_triangles: bool, output: str) -> None:
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
    tri = make_delaunay(size[1], size[0], list3, img1, img2)
    generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri, size, draw_triangles, output)


st.title("Face Morphing")

image1_path = IMAGES_DIR.joinpath("tim_profilepic.jpeg")
image2_path = IMAGES_DIR.joinpath("rih.png")

cols = st.columns(2)
with cols[0]:
    st.image(str(image1_path))
with cols[1]:
    st.image(str(image2_path))

duration = st.number_input("Morph Duration", min_value=1.0, value=5.0)
framerate = st.number_input("Morph Framerate", min_value=1, value=20)
draw_triangles = st.checkbox("Show Triangle Mesh", value=False)

if st.button("Morph!"):
    # Generate a filename
    filename = f"{uuid.uuid4()}.mp4"
    with st.spinner("Generating movie..."):
        doMorphing(
            img1=cv2.imread(str(image1_path)),
            img2=cv2.imread(str(image2_path)),
            duration=duration,
            frame_rate=framerate,
            draw_triangles=draw_triangles,
            output=filename)

    st.video(filename)
    pathlib.Path(filename).unlink(missing_ok=True)

