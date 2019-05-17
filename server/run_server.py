import sys
import os

cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd))

import cv2
import signal

import uuid
import flask
import sqlite3
import argparse

from frame_generator import FrameGenerator, StreamMode
from flask import Response, url_for, request, jsonify, render_template, send_from_directory, redirect, json, abort, send_file
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

from constants import *
from api_utilities import *


parser = argparse.ArgumentParser(description='SERVER API')
parser.add_argument('--model', type=str, default='yolo_v3',
                    help='specify object dectector to use: [faster_rcnn, yolo_v3]')

# OBJECT DECTECTOR
print_header('LOADING MODEL')
model = get_model(model_name=parser.parse_args().model)

# DATABASE
print_header('LOADING DATABASE')
connection = None
connection = create_connection(DATABASE_PATH)

if connection is not None:
    # CREATE TABLES IF NOT EXISTS
    create_table(connection, sql_create_processed_data_table)
else:
    raise Exception("Can't open database !")

# FLASK APP
print_header('LOADING FLASK APP')
app = flask.Flask(__name__)

def process(user_id, camera_id, filepath, newest_id, starting_datetime):
    # INPUT_PATH
    input_path = filepath
    filename = get_basename(filepath)

    # OUTPUT_PATH
    output_filename = get_basename(filepath)
    output_path = os.path.join(PROCESSED_DATA_PATH, output_filename.replace('h264', 'avi'))
    print(output_path)
    # process VIDEO
    if has_video_extension(filename):
        vid = cv2.VideoCapture(input_path)
        vid_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = int(vid.get(cv2.CAP_PROP_FPS) // 4)

        datetime_object = starting_datetime
        print("Start processing {}, FPS: {}, start time: {}".format(input_path, fps, datetime_object))
        output_file = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 1, vid_size)
        i = 0
        processed = 0
        while vid.isOpened():
            try:
                ret, frame = vid.read()
            except e:
                print("got trouble in decoding frame {}".format(i))
                i+=1
                continue

            i += 1
            
            if ret and i % fps == 0:
                print("Processing frame {}".format(processed + 1))
                detected, bboxes = model.predict_image(frame)

                datetime_object += timedelta(seconds=1)
                resized_frame = cv2.resize(detected, vid_size)
                output_file.write(resized_frame)
                create_processed_data_row(connection, task_info=(user_id, camera_id, datetime_object, input_path, len(bboxes)))
                processed += 1
                
                # if processed == 20:
                #     break
            if not ret:
                break

        print("Updating uploaded data status ...")
        vid.release()
        output_file.release()
        cv2.destroyAllWindows()

        update_uploaded_data_row(connection, task_info=(output_path, newest_id))
        print("Done")
        return output_path
                
@app.route('/' + PROCESS_API_PATH, methods=['POST'])
def process_data():
    if request.method == 'POST':
        print_header('A NEW REQUEST COMING !')
        # GETTING DATA
        
        # GENERATE A TASK ID
        # task_id = str(uuid.uuid4())

        # filename = f'{task_id}_{secure_filename(f.filename)}'
        file_path = request.json['file_path']
        file_id = request.json['file_id']
        print_header('GETTING DATA {}'.format(file_path))
        # filepath = os.path.join(UPLOADED_DATA_PATH, filename)
        file_name = get_basename(file_path)
        file_name = secure_filename(file_name)

        name, file_extension = os.path.splitext(file_name) # e.g: name 0_CAM1_20190302134800708
        splits = name.split('.')[0].split("_")
        user_id = int(splits[0])
        camera_id = int(splits[1][3:])

        # PROCESS THE FILE
        result = process(user_id, camera_id, file_path, file_id, interpret_name(splits[2]))
        return result

@app.route('/check_data', methods=['GET'])
def check_data():
    if request.method == 'GET':        
        params = request.form
        user_id = params['user_id']
        camera_id = params['camera_id']
        file_name = params['file_name']
        
        result = check_processed_data_row(connection, task_info=(user_id, camera_id, file_name))
        
        return json.jsonify(result)

@app.route('/get_data', methods=['GET'])
def get_data():
    if request.method == 'GET':        
        params = request.form
        user_id = params['user_id']
        camera_id = params['camera_id']
        file_name = params['file_name']
        result = get_processed_data_row(connection, task_info=(user_id, camera_id, file_name))
        print(result)
        
        return send_file(result)


@app.route('/test', methods=['POST'])
def test():
    if request.method == 'POST':        
        print(request.data)
        print(request.form)
        print(request.json)
        
        return jsonify(None)

def exit_signal_handler(sig, frame):
    print_header('CLOSE DATABASE')
    print_header('EXIT')
    if connection is not None:
        connection.close()
    sys.exit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_signal_handler)

    # app.run(debug=False, threaded=True)
    app.run(host='127.0.0.1', port=PROCESS_API_PORT, debug=False)
