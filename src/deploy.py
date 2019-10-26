from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
from keras.models import load_model


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
vocab = None

def load():
    global model, vocab

    model = load_model('model.h5')
    vocabFile = open('vocab.txt', 'r', encoding="utf8")
    vocab = [x.replace('\n', '') for x in vocabFile.readlines()]
    vocabFile.close()

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image - [103.939, 116.779, 123.68]
    # return the processed image
    return image

def predict_loop(image, max_len=661):
    encoded_equation = np.zeros((1,max_len))
    encoded_equation[0][0] = vocab[-2] # start token
    y_hat = np.expand_dims(np.array(vocab[-2]), axis=0)

    for i in range(1,max_len):
        y_hat = np.argmax(softmax(model.predict([img, y_hat]), axis=-1),axis=-1)
        encoded_equation[0][i] = yp
        if y_hat == vocab[-1]:
            break
    return encoded_equation

def decode(encoding):
    ids = np.squeeze(enc).astype(np.int32)
    tokens = [vocab[x] for x in ids]
    decoded_string = ' '.join(tokens)
    return decoded_string.strip()

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(128, 1024))
            encoding = predict_loop(image)
            equation = decode(encoding)
            data["prediction"] = equation
            data["success"] = True
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_()
    app.run()