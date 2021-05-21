import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops


def read_img(path):
    img_data = tf.io.gfile.GFile(path, "rb").read()
    img = Image.open(BytesIO(img_data))

    (img_width, img_height) = img.size

    img = np.array(img.getdata()).reshape((1, img_height, img_width, 3)).astype(np.uint8)

    return img_width, img_height, img


DIR = os.path.abspath(os.path.dirname("."))

#labels_path = os.path.join(DIR, "bin", "mscoco_label_map.pbtxt")
#category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

model_link = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1"

hub_model = hub.load(model_link)


DIR = os.path.join(DIR, "Data")

folders = os.walk(DIR).__next__()[1]

for d in folders:
    
    imgs = os.walk(os.path.join(DIR, d)).__next__()[2]

    os.makedirs(os.path.join(DIR, d, "transformed"), exist_ok=True)

    for i, img in enumerate(imgs):
        img_name = img.split(".")[0]
        
        img = os.path.join(DIR, d, img)

        img_width, img_height, img_data = read_img(img)

        results = hub_model(img_data)
        result = {key:value.numpy() for key,value in results.items()}

        # y_min, x_min, y_max, x_max
        boxes_coords = result["detection_boxes"][0]
        scores = result["detection_scores"][0]
        classes = result["detection_classes"][0]

        print("=========== IMAGE - {} - {} ===========".format(d, img))
        indexes = np.where(classes == 18.)[0]
        #index_high_score = np.argmax(scores[indexes])

        for j, index in enumerate(indexes):
            
            if scores[index] < 0.55:
                #print("Não tem acurácia suficiente: ", scores[index])
                continue

            box = boxes_coords[index]

            left, right, top, bottom = box[1] * img_width, box[3] * img_width,\
                                    box[0] * img_height, box[2] * img_height
            

            new_img = img_data[0, int(top):int(bottom), int(left):int(right), :]


            #print("Original: ", img_data.shape)
            #print("Cropped: ", new_img.shape)

            new_img = Image.fromarray(new_img, "RGB")
            
            path_transformed = os.path.join(DIR, d, "transformed")

            print("Saving...")
            if j == 0:
                new_img.save(img)
            else:
                new_img.save( os.path.join(path_transformed, "{}_{}.jpg".format(img_name, j)))