import pathlib
import uuid
import utils
import cv2
import streamlit as st
import os

from delaunay_triangulation import make_delaunay
from face_landmark_detection import generate_face_correspondences
from face_morph import generate_morph_sequence
from utils.align_images import align_images

IMAGES_DIR = pathlib.Path(__file__).parent.parent.joinpath("images/aligned_images").absolute()


def doMorphing(img1, img2, duration: float, frame_rate: int, draw_triangles: bool, output: str) -> None:
    [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
    tri = make_delaunay(size[1], size[0], list3, img1, img2)
    generate_morph_sequence(duration, frame_rate, img1, img2, points1, points2, tri, size, draw_triangles, output)


st.title("See kids grow up!")

cols = st.columns(2)
with cols[0]:
    img1uploaded = st.file_uploader("Starting image", type=["png", "jpg", "jpeg"])

with cols[1]:
    img2uploaded = st.file_uploader("Ending image", type=["png", "jpg", "jpeg"])

duration = st.number_input("Video duration", min_value=1.0, value=5.0)
framerate = st.number_input("# of frames per second", min_value=1, value=30)
draw_triangles = False

if st.button("Generate Video"):
    # Generate an unique id

    if img1uploaded is None:
        st.error("Please select a starting image")

    if img2uploaded is None:
        st.error("Please select an ending image")

    if img1uploaded is not None and img2uploaded is not None:
        key = uuid.uuid4().hex

        new_filename = f"{key}.mp4"
        img1bytes = img1uploaded.getvalue()
        with open(f"{img1uploaded.name}", 'wb') as w:
            w.write(img1bytes)
            w.close()

        img2bytes = img2uploaded.getvalue()
        with open(f"{img2uploaded.name}", 'wb') as w:
            w.write(img2bytes)
            w.close()

        with st.spinner("Aligning images..."):
            align_images(img1uploaded.name, img2uploaded.name)

        with st.spinner("Generating video..."):
            alignedimagename1 = '%s_aligned.png' % (os.path.splitext(img1uploaded.name)[0])
            alignedimagename2 = '%s_aligned.png' % (os.path.splitext(img2uploaded.name)[0])

            doMorphing(
                img1 = cv2.imread(alignedimagename1),
                img2 = cv2.imread(alignedimagename2),
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

