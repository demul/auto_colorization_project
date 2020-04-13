# Image Processing's
import numpy as np
import PIL.Image
import cv2

# Encoding's
import base64
from io import BytesIO

# Network's
import tensorflow as tf
from flask import Flask, render_template, request

# Neural Net's
import model

# Utility's
import util
import os

#####################################################################
#####################################################################

# declare model
model = model.Model(1)

# declare decoder
filename_gamut = str()
list_filenames = os.listdir('./')
for i in range(len(list_filenames)):
    if list_filenames[i][:5] == 'gamut':
        filename_gamut = list_filenames[i]
        break
    if i == len(list_filenames)-1:
        print('There is not gamut. Run \'data_preprocessor.py\' before run this script...')
decoder = util.Decoder(util.tablize_gamut(np.load(filename_gamut)))

# declare arguments
X = tf.placeholder(shape=[None, 224, 224, 1], dtype=tf.float32)
isTrain = tf.constant(False)

# make graph
logit_ = model.color_classifier(X, isTrain=isTrain)
prob_ = tf.nn.softmax(logit_)

# define session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define saver
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./ckpt')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)


def resize_crop(img):
    h = img.shape[0]
    w = img.shape[1]

    if h > w:
        img = cv2.resize(img, (224, int(h / w * 224)), interpolation=cv2.INTER_LINEAR)
    else:
        img = cv2.resize(img, (int(w / h * 224), 224), interpolation=cv2.INTER_LINEAR)

    h = img.shape[0]
    w = img.shape[1]

    return img[h // 2 - 112:h // 2 + 112, w // 2 - 112:w // 2 + 112, :]

def numpy2ascii(img):
    # numpy array to PIL Image
    img = PIL.Image.fromarray(img)

    # save image to BytesIO object
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=100)
    buffer.seek(0)

    # BytesIO object to base64 encodeing
    img_encoded = base64.b64encode(buffer.getvalue())

    # decode base64 encodeing to ascii
    img_ascii = img_encoded.decode('ascii')

    # close buffer
    buffer.close()
    return img_ascii

#####################################################################
#####################################################################


app = Flask(__name__)

@app.route('/upload')
def upload_file():
    return render_template('upload.html')

@app.route('/uploaded', methods=['GET', 'POST'])
def uploaded_file():
    if request.method == 'POST':
        # file storage object to numpy
        img_arr = np.fromfile(request.files['file'], np.uint8)

        # decode numpy to image array and resize
        img_arr = resize_crop(cv2.imdecode(img_arr, cv2.IMREAD_COLOR))

        # convert to gray(if image is already gray, isn't needed.)
        img_gray = np.expand_dims(np.mean(img_arr, axis=2), axis=2).astype(np.uint8)

        # expand both to batch
        batch_input = np.expand_dims(img_gray, axis=0) # [1, 56, 56, 1]

        # inference(CORE)
        prob = sess.run(prob_, feed_dict={X: batch_input})
        batch_output = decoder.encoding2bgr(batch_input, prob)
        img_output = np.squeeze(batch_output)
        img_output = cv2.resize(img_output, (224, 244), interpolation=cv2.INTER_LINEAR)

        img_origin = numpy2ascii(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
        img_gray = numpy2ascii(np.tile(img_gray, (1, 1, 3)))
        img_processed = numpy2ascii(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))

        return render_template('uploaded.html', image_processed=img_processed, image_gray=img_gray, image_origin=img_origin)


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=8000, debug=True)
    app.run(host='0.0.0.0', port=9000)
