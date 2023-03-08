########################################################################################################################
# KRAFTON JUNGLE 1기 나만의 무기 만들기 프로젝트
# Project Biam.io
# by.Team dabCAT
# 박찬우 : https://github.com/pcw999
# 박현우 : https://github.com/phwGithub
# 우한봄 : https://github.com/onebom
# 이민섭 : https://github.com/InFinity-dev
########################################################################################################################
##################################### PYTHON PACKAGE IMPORT ############################################################
import math
import random
# import cvzone
import cv2
import numpy as np
# import mediapipe as mp
import sys
import os
from flask_restful import Resource, Api
from flask_cors import CORS
from datetime import datetime
import datetime
import time
from flask import Flask, render_template, Response, request, redirect, url_for, session
from flask_socketio import SocketIO, emit, join_room
# import socket
from engineio.payload import Payload

from src.bgm.bgm import *
from src.handDetector.HandDetector import HandDetector
from src.game.multiGameClass import MultiGameClass
from src.game.snakeGameClass import SnakeGameClass, bot_data

# import simpleaudio as sa
import threading
import signal

# import pprint

########################################################################################################################
################################## SETTING GOLBAL VARIABLES ############################################################

Payload.max_decode_packets = 200

# PYTHON - ELECTRON VARIABLES
# This wil report the electron exe location, and not the /tmp dir where the exe
# is actually expanded and run from!
print(f"flask is running in {os.getcwd()}, __name__ is {__name__}", flush=True)
# print(f"flask/python env is {os.environ}", flush=True)
print(sys.version, flush=True)
# print(os.environ, flush=True)
# print(os.getcwd(), flush=True)
# print("User's Environment variable:")
# pprint.pprint(dict(os.environ), width = 1)

base_dir = '.'
if hasattr(sys, '_MEIPASS'):
    print('detected bundled mode', sys._MEIPASS)
    base_dir = os.path.join(sys._MEIPASS)

app = Flask(__name__, static_folder=os.path.join(base_dir, 'static'),
            template_folder=os.path.join(base_dir, 'templates'))

app.config['SECRET_KEY'] = "roomfitisdead"
app.config['DEBUG'] = True  # true will cause double load on startup
app.config['EXPLAIN_TEMPLATE_LOADING'] = False  # won't work unless debug is on

socketio = SocketIO(app, cors_allowed_origins='*')

# CORS(app, origins='http://localhost:5000')

api = Api(app)

# Setting Path to food.png
pathFood = './src-flask-server/static/food.png'

opponent_data = {}  # 상대 데이터 (현재 손위치, 현재 뱀위치)
gameover_flag = False  # ^^ 게임오버
bot_flag = False
now_my_room = ""  # 현재 내가 있는 방
now_my_sid = ""  # 현재 나의 sid
MY_PORT = 0  # socket_bind를 위한 내 포트 번호
user_number = 0  # 1p, 2p를 나타내는 번호
user_move = False
game_over_for_debug = False
start = False

# Create a thread for the BGM
bgm_thread = threading.Thread(target=play_bgm)
# Register the signal handler for SIGINT (Ctrl-C)
signal.signal(signal.SIGINT, stop_music_exit)


########################################################################################################################
################################## SNAKE GAME LOGIC SECTION ############################################################
# video setting
cap = cv2.VideoCapture(0)

# Ubuntu YUYV cam setting low frame rate problem fixed
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

cap.set(3, 1280)
cap.set(4, 720)
cap.set(cv2.CAP_PROP_FPS, 30)  # TODO : 영향 확인하기, 시간 탐지 기법 중 하나가 프레임이라 프레임 맞춰줌
fps = cap.get(cv2.CAP_PROP_FPS)

detector = HandDetector(detectionCon=0.5, maxHands=1)

########################################################################################################################
######################################## FLASK APP ROUTINGS ############################################################

game = SnakeGameClass(socketio, fps, pathFood)
multi = MultiGameClass(socketio, pathFood)
single_game = SnakeGameClass(socketio, fps, pathFood)


# Defualt Root Routing for Flask Server Check
@api.resource('/')
class HelloWorld(Resource):
    def get(self):
        print(f'Electron GET Requested from HTML', flush=True)
        data = {'Flask 서버 클라이언트 to Electron': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        return data


# Game Main Menu
@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route('/testbed')
def testbed():
    global single_game

    single_game = SnakeGameClass(socketio, fps, pathFood)
    folder_path = "./src-flask-server/static/"
    filename = "bestScore.txt"
    file_path = os.path.join(folder_path, filename)

    myBestScore = 0
    if os.path.isfile(file_path):  # check if file exists
        with open(file_path, "r") as f:
            myBestScore = int(f.read())
    else:  # create the file if it doesn't exist
        with open(file_path, "w") as f:
            f.write("0")

    single_game.bestScore = myBestScore
    # print(f"bestScore : {single_game.bestScore}")
    return render_template("testbed.html", best_score=single_game.bestScore)


@app.route('/mazerunner')
def mazerunner():
    return render_template("mazerunner.html")


# Game Screen
@app.route("/enter_snake", methods=["GET", "POST"])
def enter_snake():
    global now_my_room
    global multi

    multi = MultiGameClass(socketio, pathFood)

    now_my_room = request.args.get('room_id')
    multi.user_number = int(request.args.get('user_num'))

    return render_template("snake.html", room_id=now_my_room)


########################################################################################################################
############## SERVER SOCKET AND PEER TO PEER ESTABLISHMENT ############################################################

# 페이지에서 로컬 flask 서버와 소켓 통신 개시 되었을때 자동으로 실행
@socketio.on('connect')
def test_connect():
    print('Client connected!!!')


# 페이지에서 로컬 flask 서버와 소켓 통신 종료 되었을때 자동으로 실행
@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected!!!')


# 현재 내 포트 번호 요청
@socketio.on('my_port')
def my_port(data):
    global MY_PORT

    MY_PORT = data['my_port']


# webpage로 부터 받은 상대방 주소 (socket 통신에 사용)
@socketio.on('opponent_address')
def set_address(data):
    global MY_PORT
    global multi

    opp_ip = data['ip_addr']
    opp_port = data['port']
    sid = multi.user_number

    socketio.emit('game_ready')
    # multi.set_socket(MY_PORT, opp_ip, opp_port)
    # multi.test_connect(sid)


# socketio로 받은 상대방 정보
@socketio.on('opp_data_transfer')
def opp_data_transfer(data):
    global multi
    multi.opp_points = data['opp_body_node']


# socketio로 받은 먹이 위치
@socketio.on('set_food_location')
def set_food_loc(data):
    global multi
    multi.foodPoint = data['foodPoint']
    multi.foodOnOff = True


# socketio로 받은 먹이 위치와 상대 점수
@socketio.on('set_food_location_score')
def set_food_loc(data):
    global multi
    multi.foodPoint = data['foodPoint']
    multi.opp_score = data['opp_score']
    if multi.opp_score % 5 == 0 and multi.opp_score != 0:
        multi.opp_skill_flag = True
        sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_6_path,))
        sfx_thread.start()
        socketio.emit('warning', {'opp_skill' : 1})

    multi.foodOnOff = True


# 게임 시작
@socketio.on('game_start')
def set_start():
    global start
    start = True


@socketio.on('cutted_idx')
def set_cutted_idx(data):
    global multi
    multi.cut_idx = data['cutted_idx']
    multi.skill_length_reduction()
    multi.opp_skill_flag=False


@socketio.on("save_best")
def save_best(data):
    with open("./src-flask-server/static/bestScore.txt", "w") as f:
        # Write the new contents to the file
        f.write(data)


@socketio.on("gen_break")
def gen_break():
    global multi
    multi.gen = False


########################################################################################################################
######################################## MAIN GAME ROUNTING ############################################################
@app.route('/snake')
def snake():
    def generate():
        global multi
        global start
        start = False
        skill_cnt = 0
        opp_skill_cnt = 0

        if multi.user_number == 1:
            start_cx = 100
            start_cy = 360
            multi.previousHead = (100, 360)
        elif multi.user_number == 2:
            start_cx = 1180
            start_cy = 360
            multi.previousHead = (1180, 360)
        else:
            start_cx, start_cy = 640, 360

        while True:
            if start:
                break

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            try:
                hands = detector.findHands(img, flipType=False)
                img = detector.drawHands(img)
            except:
                hands = []

            pointIndex = []

            if hands and multi.user_move:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]
            if not multi.user_move:
                pointIndex = [start_cx, start_cy]

            if not multi.user_move:
                if multi.user_number == 1:
                    start_cx += 5
                    if start_cx > 350:
                        start_cx = 70
                        multi.user_move = True
                        multi.check_collision = True
                        multi.line_flag = True
                elif multi.user_number == 2:
                    start_cx -= 5
                    if start_cx < 930:
                        start_cx = 1210
                        multi.user_move = True
                        multi.check_collision = True
                        multi.line_flag = True

            if multi.skill_flag:
                skill_cnt += 1
                if skill_cnt % 120 == 0:
                    multi.skill_flag = False
                    skill_cnt = 0

            if multi.opp_skill_flag:
                opp_skill_cnt += 1
                if opp_skill_cnt % 120 == 0:
                    multi.opp_skill_flag = False
                    opp_skill_cnt = 0

            img = multi.update(img, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if not multi.gen:
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


########################################################################################################################
############################### TEST BED FOR GAME LOGIC DEV ############################################################

# TEST BED ROUTING
@app.route('/test')
def test():
    def generate():
        global bot_data, single_game, gameover_flag, bot_flag, user_move
        global opponent_data
        single_game.global_intialize()
        single_game.testbed_initialize()
        max_time_end = time.time() + 4
        cx, cy = 100, 360
        bot_flag = True
        user_move = False
        single_game.foodtimeLimit = time.time() + 15  # 10초 제한(앞 5초는 카운트)

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            img = detector.drawHands(img)

            if not user_move:
                cx += 5
                pointIndex = [cx, cy]
            else:
                if hands:
                    lmList = hands[0]['lmList']
                    pointIndex = lmList[8][0:2]

            single_game.bot_data_update()
            opponent_data['opp_body_node'] = bot_data["bot_body_node"]
            # print(pointIndex)

            img = single_game.update(img, pointIndex)

            # encode the image as a JPEG string∂
            _, img_encoded = cv2.imencode('.jpg', img)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            if time.time() > max_time_end:
                user_move = True
                single_game.line_flag = True

            remain_time = 0
            if user_move:
                remain_time = int(single_game.foodtimeLimit - time.time())  # 할일: html에 보내기
                # print(f"remain_time: {remain_time}")
                socketio.emit('test_timer', {"seconds": remain_time})

            if gameover_flag or (remain_time < 1 and user_move):
                print("game ended")
                gameover_flag = False
                socketio.emit('gameover')
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Main Menu Selection
@app.route('/menu_snake')
def menu_snake():
    menu_game = SnakeGameClass(socketio, fps, pathFood)

    menu_game.multi = False
    menu_game.foodOnOff = False
    menuimg = np.zeros((720, 1280, 3), dtype=np.uint8)
    menu_game.global_intialize()
    menu_game.menu_initialize()

    def generate():
        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            hands = detector.findHands(img, flipType=False)
            showimg = detector.drawHands(menuimg)
            pointIndex = []

            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            showimg = menu_game.update_blackbg(showimg, pointIndex)
            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', showimg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/maze_play')
def maze_play():
    def generate():
        global game

        game.multi = False
        game.maze_initialize()

        game.timer_end = time.time() + 91  # 1분 30초 시간제한

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)

            hands = detector.findHands(img, flipType=False)
            showimg = detector.drawHands(game.maze_img)  # 무조건 findHands 다음

            pointIndex = []
            if hands:
                lmList = hands[0]['lmList']
                pointIndex = lmList[8][0:2]

            showimg = game.update_mazeVer(showimg, pointIndex)

            # encode the image as a JPEG string
            _, img_encoded = cv2.imencode('.jpg', showimg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')

            remain_time = int(game.timer_end - time.time())  # 할일: html에 보내기
            # print(f"remain_time: {remain_time}")
            # socketio.emit('maze_timer', {"minutes": remain_time // 60, "seconds": remain_time % 60})
            socketio.emit('maze_timer', {"remain_time": remain_time})
            if remain_time < 1:
                print("game ended")
                socketio.emit('gameover')
                break

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


########################################################################################################################
########################## Legacy Electron Template Routing ############################################################
@app.route('/hello')
def hello():
    return render_template('hello.html', msg="YOU")


@app.route('/hello-vue')
def hello_vue():
    return render_template('hello-vue.html', msg="WELCOME 🌻")


########################################################################################################################
####################################### FLASK APP ARGUMENTS ############################################################

if __name__ == "__main__":
    socketio.run(app, host='localhost', port=5000, debug=False, allow_unsafe_werkzeug=True)

########################################################################################################################
