import numpy as np
import cv2
import ntpath
import sqlite3
import sys
import os

from sqlite3 import Error
from datetime import datetime
from PIL import Image


def print_header(s):
    s = s.upper().strip()

    len_header = 30

    if len(s) >= len_header:
        print(s)
    else:
        start_pos = len_header // 2 - len(s) // 2
        print(f'\n+{"-" * start_pos}{s}{"-" * (len_header-start_pos-len(s))}+\n')


def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2_img)


def pil_to_cv2(pil_img):
    return np.asarray(pil_img)


def has_video_extension(file_path):
    return file_path.lower().endswith(('.mp4', '.avi', '.h264'))


def has_image_extension(file_path):
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))


def get_basename(file_path):
    return ntpath.basename(file_path)


def interpret_name(filename="20190302134800708"):
    '''
    year = filename[:4]
    month = filename[4:6]
    day = filename[6:8]
    hour = filename[8:10]
    minute = filename[10:12]
    second = filename[12:14]
    '''

    datetime_object = datetime.strptime(filename[:14], '%Y%m%d%H%M%S')
    return datetime_object

def create_connection(db_file):
    """ create a database connection to a SQLite database """
    try:
        conn = sqlite3.connect(db_file, check_same_thread=False)
        return conn
    except Error as e:
        print(e)

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def check_existed_uploaded_data(conn, task_info):
    '''
    task_info: (user_id, camera_id, file_name)
    '''
    cur = conn.cursor()
    cur.execute("SELECT * FROM uploaded_data WHERE user_id=? AND camera_id=? AND file_name=?", task_info)
 
    rows = cur.fetchall()
    # print(rows)
    assert len(rows) <= 1, "Something went wrong, get duplicated rows which should be UNIQUE"

    return len(rows)

def get_processed_data_row(conn, task_info):
    '''
    task_info: (user_id, camera_id, file_name)
    '''
    cur = conn.cursor()
    cur.execute("SELECT * FROM uploaded_data WHERE user_id=? AND camera_id=? AND file_name=?", task_info)
 
    rows = cur.fetchall()
    assert len(rows) <= 1, "Something went wrong, get duplicated rows which should be UNIQUE"
    if len(rows) ==  0:
        return None

    return rows[0][-1]

def check_processed_data_row(conn, task_info):
    '''
    task_info: (user_id, camera_id, file_name)
    '''
    cur = conn.cursor()
    cur.execute("SELECT * FROM uploaded_data WHERE user_id=? AND camera_id=? AND file_name=?", task_info)
 
    rows = cur.fetchall()
    assert len(rows) <= 1, "Something went wrong, get duplicated rows which should be UNIQUE"
    if len(rows) ==  0:
        return False

    if rows[0][-1] is not None:
        return True

    return False

def get_uploaded_data_row(conn, task_info):
    '''
    task_info: (user_id, camera_id, file_name)
    '''
    cur = conn.cursor()
    cur.execute("SELECT * FROM uploaded_data WHERE user_id=? AND camera_id=? AND file_name=?", task_info)
 
    rows = cur.fetchall()
    assert len(rows) <= 1, "Something went wrong, get duplicated rows which should be UNIQUE"
    if len(rows) ==  0:
        return None

    return rows[0][0]

def update_uploaded_data_row(conn, task_info):
    """
    task_info: (processed_file, id)
    """
    sql = ''' UPDATE uploaded_data
              SET   processed_file = ?
              WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, task_info)
    conn.commit()

def create_uploaded_data_row(conn, task_info):
    '''
    task_info: (user_id, camera_id, file_path, file_name, processed_file)
    '''
    sql = """ INSERT INTO uploaded_data (user_id, camera_id, file_path, file_name, processed_file)
              VALUES (?,?,?,?,?); """
    cur = conn.cursor()
    cur.execute(sql, task_info)
    conn.commit()
    return cur.lastrowid


def create_processed_data_row(conn, task_info):
    '''
    task_info: (user_id, camera_id, time_stamp, file_path, counter)
    '''
    sql = """ INSERT INTO processed_data (user_id, camera_id, time_stamp, file_path, counter)
                  VALUES (?,?,?,?,?); """
    cur = conn.cursor()
    cur.execute(sql, task_info)
    conn.commit()
    return cur.lastrowid
