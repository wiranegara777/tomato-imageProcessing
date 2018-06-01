from flask import Flask, redirect, request, jsonify,render_template
import numpy as np
from PIL import Image
import io
import pickle
from sklearn.externals import joblib

import cv2
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from subprocess import check_output


app = Flask(__name__)
model = None

def load_model():
     global model
     global modelReg

     filename2 = 'reg_model.sav'
     filename = 'svm_model.sav'
     model = pickle.load(open(filename, 'rb'))
     modelReg = joblib.load('reg_model.sav')
    # modelReg = pickle.load(open(filename2, 'rb'))
    # print(loaded model)

def subrgbgray(rgb,mask):
    row, col, ch = rgb.shape  # membaca ukuran image, banyak baris, kolom, dan channel
    canvas = np.zeros((row, col, 3), np.uint8)  # membuat image kosong dengan ukuran row*col dengan 1 canal
    for i in range(0, row):  # loop baris
        for j in range(0, col):  # loop kolom
            if mask[i, j] == 0:  # kondisi substract image asli dengan image biner untuk mendapatkan segmentasi
                canvas[i, j] = rgb[i, j]  # set pixel i,j canal 0
            else:  # kondisi substract image asli dengan image biner untuk menghitamkan latar
                canvas.itemset((i, j, 0), 0)  # set pixel i,j canal 0
                canvas.itemset((i, j, 1), 0)  # set pixel i,j canal 0
                canvas.itemset((i, j, 2), 0)  # set pixel i,j canal 0
    return canvas


@app.route('/')
def index():
    return redirect('/static/index.html')

@app.route('/tomat', methods=['POST'])
def tomat():
    image = cv2.imread('test.jpg')
    blurred = cv2.GaussianBlur(image, (19, 19), 0)
    b, g, r = cv2.split(blurred)
    imgcpy = cv2.imread('test.jpg',0)
    imgcpy = cv2.medianBlur(imgcpy, 5)
    ret, mask = cv2.threshold(imgcpy, 122, 255, cv2.THRESH_BINARY)
    # masked_img = subrgbgray(image,mask)
    # ret, mask = cv2.threshold(r, 35, 255, cv2.THRESH_BINARY)
    # cv2.imshow("mask", mask)

    invert = cv2.bitwise_not(mask)
    row, col = invert.shape

    for i in range(row):
        for j in range(1300):
            invert[i][j] = 0
    mean = np.nanmean(invert)
    white = invert.sum()

    # cv2.imwrite("{}_mask.jpg".format(nama_file), invert)

    masked_img = cv2.bitwise_and(image, image, mask=invert)
    # cv2.imshow("masked image", masked_img)
    #cv2.imwrite("{}_masked.bmp".format(nama_file), masked_img)
    bgr = masked_img.copy()
    hsv = cv2.cvtColor(masked_img.copy(), cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(masked_img.copy(), cv2.COLOR_BGR2LAB)

    (bl, g, r) = cv2.split(bgr)
    (h, s, v) = cv2.split(hsv)
    (l, a, b) = cv2.split(lab)

        # cv2.imshow("hue", a)
        # --- RGB ---
    r_mean = np.nanmean(r)
    g_mean = np.nanmean(g)
    bl_mean = np.nanmean(bl)
    r_std = np.nanstd(r)
    g_std = np.nanstd(g)
    bl_std = np.nanstd(bl)

        # --- HSV --#
    h_mean = np.nanmean(h)
    s_mean = np.nanmean(s)
    v_mean = np.nanmean(v)
    h_std = np.nanstd(h)
    s_std = np.nanstd(s)
    v_std = np.nanstd(v)

        # --- Lab ---
    l_mean = np.nanmean(l)
    a_mean = np.nanmean(a)
    b_mean = np.nanmean(b)
    l_std = np.nanstd(l)
    a_std = np.nanstd(a)
    b_std = np.nanstd(b)

    data = [
         # "berat": folder_name,
            #"filename": file,
            r_mean,
            g_mean,
            bl_mean,
            h_mean,
            s_mean,
            v_mean,
            l_mean,
            a_mean,
            b_mean,
            r_std,
            g_std,
            bl_std,
            h_std,
            s_std,
            v_std,
            l_std,
            a_std,
            b_std,
        ]
    data2 = [
        mean, white
    ]
    #confidence = str(round(max(pred[0]), 3))
    pred =  model.predict([data])
    if(pred[0] == 1):
        kelas = "Mentah"
    elif(pred[0] == 2):
        kelas = "Setengah Matang"
    elif(pred[0] == 3):
        kelas = "Cukup Matang"
    elif(pred[0] == 4):
        kelas = "Matang"
    elif(pred[0] == 5):
        kelas = "Sangat Matang"
    predReg = modelReg.predict([data2])
    answer = str(round(predReg[0][0], 2))

    data = dict(pred=pred[0], confidence=predReg[0][0])
    # return jsonify(data)
    return render_template('result.html', prediction = kelas, weight=answer)
    #return redirect('/templates/result.html', prediction = pred[0], weight=predReg[0][0])

@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('test.jpg')
    #     image = cv2.imread('test.jpg')
    #     blurred = cv2.GaussianBlur(image, (19, 19), 0)
    #     b, g, r = cv2.split(blurred)
    #     imgcpy = cv2.imread('test.jpg',0)
    #     imgcpy = cv2.medianBlur(imgcpy, 5)
    #     ret, mask = cv2.threshold(imgcpy, 122, 255, cv2.THRESH_BINARY)
    #     # masked_img = subrgbgray(image,mask)
    #     # ret, mask = cv2.threshold(r, 35, 255, cv2.THRESH_BINARY)
    #     # cv2.imshow("mask", mask)
    #
    #     invert = cv2.bitwise_not(mask)
    #     row, col = invert.shape
    #
    #     for i in range(row):
    #         for j in range(1300):
    #             invert[i][j] = 0
    #     mean = np.nanmean(invert)
    #     white = invert.sum()
    #
    #     # cv2.imwrite("{}_mask.jpg".format(nama_file), invert)
    #
    #     masked_img = cv2.bitwise_and(image, image, mask=invert)
    #     # cv2.imshow("masked image", masked_img)
    #     #cv2.imwrite("{}_masked.bmp".format(nama_file), masked_img)
    #     bgr = masked_img.copy()
    #     hsv = cv2.cvtColor(masked_img.copy(), cv2.COLOR_BGR2HSV)
    #     lab = cv2.cvtColor(masked_img.copy(), cv2.COLOR_BGR2LAB)
    #
    #     (bl, g, r) = cv2.split(bgr)
    #     (h, s, v) = cv2.split(hsv)
    #     (l, a, b) = cv2.split(lab)
    #
    #         # cv2.imshow("hue", a)
    #         # --- RGB ---
    #     r_mean = np.nanmean(r)
    #     g_mean = np.nanmean(g)
    #     bl_mean = np.nanmean(bl)
    #     r_std = np.nanstd(r)
    #     g_std = np.nanstd(g)
    #     bl_std = np.nanstd(bl)
    #
    #         # --- HSV --#
    #     h_mean = np.nanmean(h)
    #     s_mean = np.nanmean(s)
    #     v_mean = np.nanmean(v)
    #     h_std = np.nanstd(h)
    #     s_std = np.nanstd(s)
    #     v_std = np.nanstd(v)
    #
    #         # --- Lab ---
    #     l_mean = np.nanmean(l)
    #     a_mean = np.nanmean(a)
    #     b_mean = np.nanmean(b)
    #     l_std = np.nanstd(l)
    #     a_std = np.nanstd(a)
    #     b_std = np.nanstd(b)
    #
    #     data = [
    #          # "berat": folder_name,
    #             #"filename": file,
    #             r_mean,
    #             g_mean,
    #             bl_mean,
    #             h_mean,
    #             s_mean,
    #             v_mean,
    #             l_mean,
    #             a_mean,
    #             b_mean,
    #             r_std,
    #             g_std,
    #             bl_std,
    #             h_std,
    #             s_std,
    #             v_std,
    #             l_std,
    #             a_std,
    #             b_std,
    #         ]
    #     data2 = [
    #         mean, white
    #     ]
    #     #confidence = str(round(max(pred[0]), 3))
    #     pred =  model.predict([data])
    #     predReg = modelReg.predict([data2])
    #
    #     data = dict(pred=pred[0], confidence=predReg[0][0])
    #     return jsonify(data)
    #
    # return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    fileob = open('test.jpg', 'rb')
    data = fileob.read()
    return data


if __name__ == '__main__':
    load_model()
    #predict()
    # model._make_predict_function()
    app.run(debug=False, port=5000)
