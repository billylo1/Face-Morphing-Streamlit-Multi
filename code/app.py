import pathlib
import uuid
import utils
import cv2
import streamlit as st
import os

from delaunay_triangulation import make_delaunay
from face_landmark_detection import generate_face_correspondences
from face_morph import generate_morph_sequence
from utils.align_images import align_image
from subprocess import Popen, PIPE

IMAGES_DIR = pathlib.Path(__file__).parent.parent.joinpath("images/aligned_images").absolute()


def doMorphing(alignedimagenames, duration: float, frame_rate: int, draw_triangles: bool, output: str) -> None:
    
    img1 = None
    img2 = None
    p = None

    num_images = int((duration*frame_rate)/(alignedimagenames.__len__()-1))

    for imgname in alignedimagenames:
        if not pathlib.Path(imgname).is_file():
            raise FileNotFoundError(f"File {imgname} not found")
        if img1 is None:
            img1 = cv2.imread(imgname)
            continue
        else:
            img2 = cv2.imread(imgname)
            [size, img1, img2, points1, points2, list3] = generate_face_correspondences(img1, img2)
            if p is None:
                p = Popen(
                    ['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(frame_rate), '-s', str(size[1]) + 'x' + str(size[0]), '-i', '-',
                    '-c:v', 'libx264', '-crf', '25', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', output],
                    stdin=PIPE)
            tri = make_delaunay(size[1], size[0], list3, img1, img2)
            generate_morph_sequence(num_images, img1, img2, points1, points2, tri, size, draw_triangles, p.stdin )
            img1 = img2
    
    p.stdin.close()
    p.wait()

st.set_page_config(
    page_title="magic.billylo.ca",
    page_icon="🧒")

st.title("See kids grow up!")
st.header("Upload your images, and we'll generate a video transitioning from one to the next, ordered by filenames")
uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

duration = st.number_input("Video duration", min_value=1.0, value=4.0)
framerate = st.number_input("# of frames per second", min_value=1, value=20)
draw_triangles = False

if st.button("Generate Video"):
    # Generate an unique id

    if uploaded_files is None or uploaded_files.__len__() < 2:
        st.error("Please include at least two images")

    key = uuid.uuid4().hex
    new_filename = f"{key}.mp4"
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)

    with st.spinner("Aligning images..."):

        alignedimagenames = []
        for uploaded_file in uploaded_files:
            imgbytes = uploaded_file.getvalue()
            with open(f"{uploaded_file.name}", 'wb') as w:
                w.write(imgbytes)
                w.close()
            align_image(uploaded_file.name)
            alignedimagename = '%s_aligned.png' % (os.path.splitext(uploaded_file.name)[0])
            # print(alignedimagename)
            alignedimagenames.append(alignedimagename)

    with st.spinner("Generating video..."):

        doMorphing(
            alignedimagenames=alignedimagenames,
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

