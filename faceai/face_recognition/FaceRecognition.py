import dlib
import cv2
import numpy
import sys
sys.path.append("..")
from tool import compare, imtool, save_load


class FaceRecognition:
    def __init__(self, shape_predict_path, recognition_model_path):
        self.shape_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predict_path)
        self.face_recognition = dlib.face_recognition_model_v1(recognition_model_path)
        self.train_data = []
        self.train_label = []

    def train(self, train_data, is_debug=False):
        if not isinstance(train_data, list):
            raise TypeError("training_data required to be a list which contains labels and images")

        self.train_data = []
        self.train_label = []
        for data in train_data:
            if is_debug:
                print("Processing {}".format(data.get("label")))
            for img in data.get("data"):
                encodings, bounds = self._get_face_encoding(img)
                self.train_data = self.train_data + encodings
                self.train_label = self.train_label + [data.get("label") for i in encodings]

        return self

    def _get_face_encoding(self, image):
        bounds = []
        encodings = []
        detect_faces = self.shape_detector(image, 1)
        for face in detect_faces:
            shape = self.shape_predictor(image, face)
            encodings.append(self.face_recognition.compute_face_descriptor(image, shape))
            bounds.append((face.left(), face.top(), face.right(), face.bottom()))

        return encodings, bounds

    def predict(self, image, threshold=0.6):
        encodings, bounds = self._get_face_encoding(image)
        predict_info = []
        for i, vector in enumerate(encodings):
            sort_indexes, distances = compare.compare(self.train_data, vector)
            distance = distances[sort_indexes[0]]
            if distance < threshold:
                predict_info.append((self.train_label[sort_indexes[0]], bounds[i]))
            else:
                predict_info.append((None, bounds[i]))

        return predict_info

    def save(self, file_name):
        data = {
            "train_data": self.train_data,
            "train_label": self.train_label
        }
        save_load.save(data, file_name)
        return self

    def load(self, file_name):
        data = save_load.load(file_name)
        self.train_data = data.get("train_data")
        self.train_label = data.get("train_label")
        return self


if __name__ == '__main__':
    import sys
    sys.path.append("..")
    from tool import file

    recognition_path= "../dlib_model/dlib_face_recognition_resnet_model_v1.dat"
    shape_path = "../dlib_model/shape_predictor_5_face_landmarks.dat"

    train_data = file.get_images_data("../../images/train")
    test_data = file.get_images_data("../../images/test")

    face = FaceRecognition(shape_path, recognition_path)
    face.load("param.txt")
    # face.train(d, is_debug=True)
    # face.save("param.txt")

    for data in test_data:
        for image in data.get("data"):
            for info in face.predict(image):
                imtool.put_text(image, info[0] if info[0] is not None else "Unknown", info[1])
                imtool.draw_rect(image, info[1])
                imtool.show_image(image, width=400)
