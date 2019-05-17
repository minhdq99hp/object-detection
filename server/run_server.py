import sys
# sys.path.append('/home/minhdq99hp/iot-camera')
# sys.path.append('/home/minhdq99hp/iot-camera/minh_custom_keras_yolo3')

import os
import cv2

import models.yolo3.YOLO_v3.Yolo3 as Yolo3
import tensorflow as tf


import uuid
import flask
from flask import Response, request, jsonify, render_template, send_from_directory, redirect, json
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

from api_utilities.util import print_header, pil_to_cv2, cv2_to_pil


# CONSTANT
processed_data_path = "database/processed_data"
uploaded_data_path = "database/uploaded_data"
show_frame = False

yolo = None
graph = None

print_header('LOADING FLASK APP')
app = flask.Flask(__name__)

Bootstrap(app)

def process(filename, output_type):
    with graph.as_default():
        # INPUT_PATH
        input_path = os.path.join(uploaded_data_path, filename)

        # OUTPUT_PATH
        output_filename = f'{str(uuid.uuid4())}_{filename}'
        output_path = os.path.join(processed_data_path, output_filename)

        # OUTPUT_INFO
        detection_info = {'frames': [],
                          'time_interval': 0,
                          'count_frames': 0,
                          'output_path': output_path,
                          'output_filename': output_filename}


        # process IMAGE
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_img = cv2.imread(input_path)

            detected, frame_info = yolo.detect_person_cv2(input_img)

            cv2.imwrite(os.path.join(processed_data_path, output_filename), detected)

            detection_info['frames'].append(frame_info)
            detection_info['count_frames'] += 1
            detection_info['time_interval'] += frame_info['time_interval']

        # process VIDEO
        elif filename.lower().endswith(('.mp4', '.avi')):
            # USING FRAME GENERATOR
            frame_generator = FrameGenerator(StreamMode.VIDEO, input_path)

            output_file = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                          frame_generator.vid_fps, frame_generator.vid_size)

            for frame in frame_generator.yield_frame():
                detected, frame_info = yolo.detect_person_cv2(frame)

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


# def yolo.detect_person_cv2(img):
#     pil_im = cv2_to_pil(img)
#
#     with graph.as_default():
#         detected, detection_info = yolo.detect_person_cv2(pil_im)
#
#     detected = pil_to_cv2(detected)
#
#     return detected, detection_info


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


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/upload_data', methods=['GET', 'POST'])
def upload_data():
    if request.method == 'POST':
        print_header('A NEW REQUEST COMING !')
        # GETTING DATA
        print_header('GETTING DATA')
        f = request.files['file_input']
        filename = secure_filename(f.filename)
        output_type = request.form.get("output_type")

        # SAVE FILE TO LOCAL
        print_header('SAVE FILE TO LOCAL')
        filepath = os.path.join(uploaded_data_path, filename)
        f.save(filepath)

        # process THE FILE
        print_header('process FILE')

        # TODO: change this line using celery
        # task = process.apply_async(args=[filename, output_type])
        result = process(filename, output_type)

        output_file_path = os.path.join(processed_data_path, result['output_filename'])

        if output_type == 'output_file':  # RETURN processED FILE
            try:
                print_header('RETURN processED FILE')
                # return send_file(output_file_path, attachment_filename=result["output_filename"])
                return send_from_directory(processed_data_path, result['output_filename'])
            except Exception as e:
                print(e)
        elif output_type == 'output_json':  # RETURN JSON FILE
            return jsonify(result)

        else:  # RETURN HTML PAGE
            print_header('OUTPUT_HTML')
            if output_file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # return render_template('index.html',
                #                        result_file=Markup(f'<img class="img-fuild" '
                #                                           f'style="max-width:100%; height:auto;" '
                #                                           f'src="{output_file_path}" alt="Result">'))
                return redirect(output_file_path)
            elif output_file_path.lower().endswith(('.mp4', '.avi')):
                # return render_template('index.html',
                #                        result_file=Markup(f'<video style="max-width:100%; height:auto;" controls>'
                #                                           f'<source src="{output_file_path}" type="video/mp4">'
                #                                           f'Sorry, your browser doesn\'t support embedded videos.'
                #                                           f'</video>'))
                return redirect(output_file_path)


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
