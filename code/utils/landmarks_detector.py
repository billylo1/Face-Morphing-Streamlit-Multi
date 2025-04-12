import dlib

class LandmarksDetector:
    def __init__(self, predictor_model_path='code/utils/shape_predictor_68_face_landmarks.dat'):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        print("Loading landmarks detector...")
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        print("Getting rgb...")
        img = dlib.load_rgb_image(image)
        print("Loaded image:")
        dets = self.detector(img, 1)
        print("Detected faces:")
        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except:
                print("Exception in get_landmarks()!")