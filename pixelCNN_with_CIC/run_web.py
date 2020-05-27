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
model = model.PixelCNN(1)

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
luminance_ = tf.placeholder(shape=[None, 224, 224, 1], dtype=tf.float32)
chrominance_ = tf.placeholder(shape=[None, 28, 28, 2], dtype=tf.float32)
isTrain = tf.constant(False)

# make graph
logit_ = model.colorizer(luminance_, chrominance_, isTrain=isTrain)
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

def recursive_image_generation(sess__, L__, lb_ab__, isTrain__, prob___, input_batch__, label_ab_batch__):
    # recursive multimodal sampling
    result_img = np.zeros([1, 28, 28, 2], dtype=np.uint8)

    ## 'only a first pixel' is picked from label_ab_batch!
    ## and rest pixels are sampled recursively
    result_img[:, 0, 0, :] = label_ab_batch__[:, 0, 0, :] # first (0, 0) index of result
    prob____ = sess__.run(prob___, feed_dict={L__: input_batch__, lb_ab__: label_ab_batch__, isTrain__: False})
    result_img[:, 0, 1, :] = decoder.encoding2ab(prob____[:, 0, 1, :]) # second (0, 1) index of result

    ## make index list [(0, 2), (0, 3).....(27, 26), (27, 27)]
    list_idx_order = []
    for ii in range(28):
        for jj in range(28):
            if jj == 0 or jj == 1:
                continue
            list_idx_order.append([ii, jj])

    for idx_pair in list_idx_order:
        prob____ = sess.run(prob___, feed_dict={L__: input_batch__, lb_ab__: result_img, isTrain__: False})
        result_img[:, idx_pair[0], idx_pair[1], :] = decoder.encoding2ab(
            prob____[:, idx_pair[0], idx_pair[1], :])
    return result_img

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
        batch_input = np.expand_dims(img_gray, axis=0) # [1, 224, 224, 1]

        # inference(CORE)
        output_batch_ab = recursive_image_generation(sess, luminance_, chrominance_, isTrain, prob_,
                                                     batch_input, np.zeros([1, 28, 28, 2], dtype=np.float32))
        output_batch = util.Lab2bgr(batch_input, output_batch_ab)
        # [1, 224, 224, 3]
        img_output = np.squeeze(output_batch)
        # [224, 224, 3]

        img_origin = numpy2ascii(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
        img_gray = numpy2ascii(np.tile(img_gray, (1, 1, 3)))
        img_processed = numpy2ascii(cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB))

        return render_template('uploaded.html', image_processed=img_processed, image_gray=img_gray, image_origin=img_origin)


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=8000, debug=True)
    app.run(host='0.0.0.0', port=9000)
