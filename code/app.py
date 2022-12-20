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
    # st.image(str(image1_path))
    img1uploaded = st.file_uploader("Starting image", type=["png", "jpg", "jpeg"])

with cols[1]:
    # st.image(str(image2_path))
    img2uploaded = st.file_uploader("Ending image", type=["png", "jpg", "jpeg"])

duration = st.number_input("Morph Duration", min_value=1.0, value=5.0)
framerate = st.number_input("Morph Framerate", min_value=1, value=20)
draw_triangles = st.checkbox("Show Triangle Mesh", value=False)

if st.button("Morph!"):
    # Generate a filename
    new_filename = f"{uuid.uuid4()}.mp4"
    img1bytes = img1uploaded.getvalue()
    with open(img1uploaded.name, 'wb') as w:
        w.write(img1bytes)
        w.close()

    img2bytes = img2uploaded.getvalue()
    with open(img2uploaded.name, 'wb') as w:
        w.write(img2bytes)
        w.close()

    with st.spinner("Generating movie..."):
        doMorphing(
            # img1=cv2.imread(str(image1_path)),
            # img2=cv2.imread(str(image2_path)),
            img1 = cv2.imread(img1uploaded.name),
            img2 = cv2.imread(img2uploaded.name),
            duration=duration,
            frame_rate=framerate,
            draw_triangles=draw_triangles,
            output=new_filename)

        # Delete the previous movie, if we have one
        movie_filename = st.session_state.get("movie_filename")
        if movie_filename is not None:
            try:
                pathlib.Path(movie_filename).unlink(missing_ok=True)
            except:
                pass

        st.session_state["movie_filename"] = new_filename

movie_filename = st.session_state.get("movie_filename")
if movie_filename is not None and pathlib.Path(movie_filename).is_file():
    st.video(movie_filename)

