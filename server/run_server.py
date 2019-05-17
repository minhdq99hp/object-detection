import sys
import os
from os.path import realpath, dirname
sys.path.append(dirname(realpath(__file__)))

import cv2

from models.yolo3.YOLO_v3 import Yolo3
import tensorflow as tf


import uuid
import flask
from flask import Response, request, jsonify, render_template, send_from_directory, redirect, json
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from utilities.util import *
from utilities.frame_generator import FrameGenerator, StreamMode

# CONSTANT
processed_data_path = "database/processed_data"
uploaded_data_path = "database/uploaded_data"
show_frame = False

yolo = None
graph = None

print_header('LOADING FLASK APP')
app = flask.Flask(__name__)

Bootstrap(app)


def generate_output_filepath(filepath):
    return f'{processed_data_path}/{str(uuid.uuid4())}_{os.path.basename(filepath)}'


def process(input_path):
    # OUTPUT_PATH
    output_path = generate_output_filepath(input_path)

    print(f'OUTPUT_PATH: {output_path}')

    # OUTPUT_INFO
    detection_info = {'frames': [],
                      'time_interval': 0,
                      'count_frames': 0,
                      'output_path': output_path
                      }

    # PROCESS IMAGE
    if has_image_extension(input_path):

        print(f'INPUT_PATH: {input_path}')

        input_img = cv2.imread(input_path)

        detected, frame_info = yolo.predict_image(input_img)

        cv2.imwrite(output_path, detected)

        detection_info['frames'].append(frame_info)
        detection_info['count_frames'] += 1
        detection_info['time_interval'] += frame_info['time_interval']

    # process VIDEO
    elif has_video_extension(input_path):
        # USING FRAME GENERATOR
        frame_generator = FrameGenerator(StreamMode.VIDEO, input_path)

        output_file = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                      frame_generator.vid_fps, frame_generator.vid_size)

        for frame in frame_generator.yield_frame():
            detected, frame_info = yolo.predict_image(frame)

            if show_frame:
                cv2.imshow('detected', detected)

            output_file.write(detected)

            detection_info['frames'].append(frame_info)
            detection_info['time_interval'] += frame_info['time_interval']
            detection_info['count_frames'] += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        output_file.release()
        cv2.destroyAllWindows()

    return detection_info

# # detection_info (of a single frame):
# time_interval
# count_boxes
# output_filename
# boxes
#   label
#   box
#   score

# detection_info (of a video):
# time_interval
# count_frames
# frames


def load_models():
    global yolo
    # global graph
    yolo = Yolo3()
    # graph = tf.get_default_graph()


def clean_static_folder():
    # VERY DANGEROUS
    if len(os.listdir(processed_data_path)) > 0:
        os.system(f'rm {os.path.join(processed_data_path, "*")}')

    if len(os.listdir(uploaded_data_path)) > 0:
        os.system(f'rm {os.path.join(uploaded_data_path, "*")}')


# @app.route('/', methods=['GET'])
# def index():
#     return render_template("index.html")


@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        print(request)

        print_header('A NEW REQUEST COMING !')
        # GETTING DATA
        print_header('GETTING DATA')
        f = request.files['input_file']
        filename = secure_filename(f.filename)

        # SAVE FILE TO LOCAL
        print_header('SAVE FILE TO LOCAL')
        filepath = realpath(os.path.join(uploaded_data_path, filename))
        f.save(filepath)

        # PROCESS THE FILE
        print_header('PROCESS FILE')
        result = process(filepath)

        # output_file_path = os.path.join(processed_data_path, result['output_filename'])

        return jsonify(result)


# def process_webcam_streaming(frame_generator, streaming_id):
#
#     for frame in frame_generator.yield_frame():
#
#         filename = f'{streaming_id}.jpg'
#         filepath = os.path.join(processed_data_path, filename)
#
#         detected, frame_info = yolo.detect_person_cv2(frame)
#
#         cv2.imwrite(filepath, detected)
#
#         binary_file = open(filepath, 'rb').read()
#
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + binary_file + b'\r\n')
#
#
# def process_video_streaming(frame_generator, streaming_id):
#     filename = f'{streaming_id}.jpg'
#     filepath = os.path.join(processed_data_path, filename)
#
#
#     for frame in frame_generator.yield_frame():
#         detected, frame_info = yolo.detect_person_cv2(frame)
#
#         cv2.imwrite(filepath, detected)
#         if show_frame:
#             cv2.imshow("Frame", detected)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#         binary_file = open(filepath, 'rb').read()
#
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + binary_file + b'\r\n')
#
#     cv2.destroyAllWindows()
#
#
# @app.route('/streaming', methods=['GET', 'POST'])
# def streaming():
#     if request.method == 'GET':
#
#         data = request.get_json()
#         print(data)
#
#         if data is None:
#             streaming_id = uuid.uuid4()
#             # TEST ON WEBCAM
#             print_header('START STREAMING WEBCAM')
#             frame_generator = FrameGenerator(StreamMode.WEBCAM)
#
#         elif 'file_path' in data:
#             streaming_id = data['streaming_id']
#             # START STREAMING VIDEO
#             print_header('START STREAMING VIDEO')
#
#             # GETTING UPLOADED FILE
#             filepath = ""
#             for p in os.listdir(uploaded_data_path):
#                 if p.startswith(streaming_id):
#                     filepath = os.path.join(uploaded_data_path, p)
#                     break
#             if filepath == "":
#                 raise Exception('Can\'t find the streaming file !')
#
#             frame_generator = FrameGenerator(StreamMode.VIDEO, filepath)
#
#         elif 'rtsp_url' in request.args:
#             print_header('START STREAMING RTSP')
#             rtsp_url = request.args.get('rtsp_url')
#
#             streaming_id = request.args['streaming_id']
#             # START STREAMING RTSP
#
#             frame_generator = FrameGenerator(StreamMode.RTSP, rtsp_url)
#
#         else:
#             streaming_id = str(uuid.uuid4())
#             frame_generator = FrameGenerator(StreamMode.WEBCAM)
#
#             return Response(process_webcam_streaming(frame_generator, streaming_id),
#                             mimetype='multipart/x-mixed-replace; boundary=frame')
#
#         return Response(process_video_streaming(frame_generator, streaming_id),
#                         mimetype='multipart/x-mixed-replace; boundary=frame')
#
#     elif request.method == 'POST':
#         result = {}
#
#         print_header('A NEW REQUEST COMING !')
#         # GENERATING A NEW STREAMING ID
#         streaming_id = uuid.uuid4()
#         result['streaming_id'] = streaming_id
#
#         if 'file_input' in request.files:
#             # GETTING DATA
#             print_header('GETTING DATA')
#             f = request.files['file_input']
#             filename = secure_filename(f.filename)
#
#             # SAVE FILE TO LOCAL. Filename = streaming_id + extension
#             print_header('SAVE FILE TO LOCAL')
#             filepath = os.path.join(uploaded_data_path, f'{streaming_id}{os.path.splitext(filename)[1]}')
#             f.save(filepath)
#
#             result['file_path'] = filepath
#
#         elif 'rtsp_url' in request.form:
#             result['rtsp_url'] = request.form['rtsp_url']
#
#
#         return jsonify(result)


if __name__ == "__main__":
    print_header('LOADING YOLO')
    load_models()

    print_header('CLEAN STATIC FOLDER')
    clean_static_folder()

    app.run(debug=False, threaded=True)
