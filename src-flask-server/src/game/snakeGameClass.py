
import numpy as np
import math, random, time
import socket
import threading
import cvzone
import cv2

from src.bgm.bgm import *
from src.maze.maze_manager import MazeManager

# Color templates
red = (0, 0, 255)  # red
megenta = (255, 0, 255)  # magenta
green = (0, 255, 0)  # green
yellow = (0, 255, 255)  # yellow
cyan = (255, 255, 0)  # cyan

bot_data = {'bot_head_x': 1000,
            'bot_head_y': 360,
            'bot_body_node': [],
            'currentLength': 0,
            'lengths': [],
            'bot_velocityX': random.choice([-1, 1]),
            'bot_velocityY': random.choice([-1, 1])}
bot_cnt=0

class SnakeGameClass:
    # 생성자, class를 선언하면서 기본 변수들을 설정함
    def __init__(self, socketio, fps, pathFood):
        self.socketio=socketio
        self.fps=fps
        
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = (int), (int)  # TODO 이거 됨 ?

        self.speed = 5
        self.minspeed = 10
        self.maxspeed = math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360

        self.score = 0
        self.bestScore = 0

        self.opp_score = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0
        self.foodOnOff = True
        self.multi = False

        self.maze_start = [[], []]
        self.maze_end = [[], []]
        self.maze_map = np.array([])
        self.passStart = False
        self.passMid = False
        self.maze_img = np.array([0])
        self.dist = 500

        self.menu_type = 0
        self.menu_time = 0
        self.line_flag = False

    def global_intialize(self):
        global user_number
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = (int), (int)  # TODO 이거 됨 ?

        self.speed = 5
        self.minspeed = 10
        self.maxspeed = math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360
        self.foodtimeLimit = 0

        self.score = 0
        self.opp_score = 0
        self.opp_addr = ()
        self.is_udp = False
        self.udp_count = 0
        self.foodOnOff = True
        self.multi = False

        self.timer_end = 0
        self.maze_start = [[], []]
        self.maze_end = [[], []]
        self.maze_map = np.array([])
        self.passStart = False
        user_number = 0

    def ccw(self, p, a, b):
        s = p[0] * a[1] + a[0] * b[1] + b[0] * p[1]
        s -= (p[1] * a[0] + a[1] * b[0] + b[1] * p[0])

        if s > 0:
            return 1
        elif s == 0:
            return 0
        else:
            return -1

    def segmentIntersects(self, p1_a, p1_b, p2_a, p2_b):
        ab = self.ccw(p1_a, p1_b, p2_a) * self.ccw(p1_a, p1_b, p2_b)
        cd = self.ccw(p2_a, p2_b, p1_a) * self.ccw(p2_a, p2_b, p1_b)

        if (ab == 0 and cd == 0):
            if (p1_a[0] > p1_b[0] or p1_a[1] > p1_b[1]):
                p1_a, p1_b = p1_b, p1_a
            if (p2_a[0] > p2_b[0] or p2_a[1] > p2_b[1]):
                p2_a, p2_b = p2_b, p2_a
            return (p2_a[0] <= p1_b[0] and p2_a[1] <= p1_b[1]) and (p1_a[0] <= p2_b[0] and p1_a[1] <= p2_b[1])

        return ab <= 0 and cd <= 0

    def isCollision(self, u1_head_pt, u2_pts):
        if not u2_pts:
            return False
        # p1_a, p1_b = np.array(u1_head_pt[0]), np.array(u1_head_pt[1]) # p1_b: head point
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for u2_pt in u2_pts:
            # p2_a, p2_b = np.array(u2_pt[0]), np.array(u2_pt[1])
            p2_a, p2_b = u2_pt[0], u2_pt[1]

            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                # print(p1_a, p1_b, p2_a, p2_b)
                return True

        return False

    def maze_collision(self, head_pt, previous_pt):
        head_pt = np.array(head_pt).astype(int)
        # if self.maze_map[int(head_pt[1]),int(head_pt[0])]==1:
        #   return True
        pt_a = np.array(previous_pt).astype(int)
        line_norm = np.linalg.norm(pt_a - head_pt).astype(int)
        points_on_line = np.linspace(pt_a, head_pt, line_norm)
        for p in points_on_line:
            try:
                if self.maze_map[int(p[1]), int(p[0])] == 1:
                    return True
            except:
                pass

        return False
    
    def create_maze(self, image_h, image_w, block_rows, block_cols):
        manager = MazeManager()
        maze = manager.add_maze(block_rows, block_cols)
        manager.solve_maze(maze.id, "DepthFirstBacktracker")

        wall_map = np.zeros((image_h, image_w))  # (h,w)
        block_h = image_h // block_rows
        block_w = image_w // block_cols

        start = [[], []]
        end = [[], []]
        r = 2

        for i in range(block_rows):
            for j in range(block_cols):
                if maze.initial_grid[i][j].is_entry_exit == "entry":
                    start = [[j * block_w + 150, i * block_h + 150], [(j + 1) * block_w + 150, (i + 1) * block_h + 150]]
                    wall_map[i * block_h + 10: (i + 1) * block_h - 10, j * block_w + 10: (j + 1) * block_w - 10] = 2
                    # print(f"start in create_maze: {start}")
                elif maze.initial_grid[i][j].is_entry_exit == "exit":
                    end = [[j * block_w + 150, i * block_h + 150], [(j + 1) * block_w + 150, (i + 1) * block_h + 150]]
                    wall_map[i * block_h + 10: (i + 1) * block_h - 10, j * block_w + 10: (j + 1) * block_w - 10] = 3
                    # print(f"end in create_maze:{end}")
                if maze.initial_grid[i][j].walls["top"]:
                    if i == 0:
                        wall_map[i * block_h:i * block_h + r, j * block_w:(j + 1) * block_w] = 1
                    else:
                        wall_map[i * block_h - r:i * block_h + r, j * block_w:(j + 1) * block_w] = 1
                if maze.initial_grid[i][j].walls["right"]:
                    wall_map[i * block_h:(i + 1) * block_h, (j + 1) * block_w - r:(j + 1) * block_w + r] = 1
                if maze.initial_grid[i][j].walls["bottom"]:
                    wall_map[(i + 1) * block_h - r:(i + 1) * block_h + r, j * block_w:(j + 1) * block_w] = 1
                if maze.initial_grid[i][j].walls["left"]:
                    if j == 0:
                        wall_map[i * block_h:(i + 1) * block_h, j * block_w:j * block_w + r] = 1
                    else:
                        wall_map[i * block_h:(i + 1) * block_h, j * block_w - r:j * block_w + r] = 1

        solution_nodes = maze.solution_path
        mid_goal_h = maze.solution_path[-3][0][0]  # solution path의 출구로부터 2번쨰 노드
        mid_goal_w = maze.solution_path[-3][0][1]
        # print(len(solution_nodes))
        mid = [[mid_goal_w * block_w + 150, mid_goal_h * block_h + 150],
            [(mid_goal_w + 1) * block_w + 150, (mid_goal_h + 1) * block_h + 150]]
        # wall_map[mid_goal_h * block_h : (mid_goal_h + 1) * block_h , mid_goal_w * block_w :(mid_goal_w + 1) * block_w] = 4

        return start, mid, end, wall_map

    # maze 초기화
    def maze_initialize(self):
        global bot_flag
        bot_flag = False
        self.maze_start, self.maze_mid, self.maze_end, self.maze_map = self.create_maze(720 - 300, 1280 - 300, 5, 12)
        self.maze_map = np.pad(self.maze_map, ((150, 150), (150, 150)), 'constant', constant_values=0)
        self.maze_img = self.create_maze_image()

        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.points = []
        self.maxspeed = math.hypot(1280, 720) / 10
        self.passStart = False
        self.passMid = False
        self.line_flag = True
        self.timer_end = time.time() + 91

    def menu_initialize(self):
        global bot_flag
        bot_flag = False
        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.line_flag = True
        self.points = []

    def testbed_initialize(self):
        global bot_data
        self.previousHead = (0, 360)
        self.velocityX = 0
        self.velocityY = 0
        self.points = []
        self.foodOnOff = True
        self.multi = False

        bot_data = {'bot_head_x': 1000,
                    'bot_head_y': 360,
                    'bot_body_node': [],
                    'currentLength': 0,
                    'lengths': [],
                    'bot_velocityX': random.choice([-1, 1]),
                    'bot_velocityY': random.choice([-1, 1])}

    def draw_snakes(self, imgMain, points, HandPoints, isMe):
        global bot_flag

        bodercolor = cyan
        maincolor = red

        if isMe:
            bodercolor = megenta
            maincolor = green
            # Draw Score
            # cvzone.putTextRect(imgMain, f'Score: {score}', [0, 40],
            #                    scale=3, thickness=3, offset=10)

        # Change hue every 100ms
        change_interval = 100

        hue = int(time.time() * change_interval % 180)  # TODO : 마지막에 성능 부족 시 아낄 수 있음
        rainbow = np.array([hue, 255, 255], dtype=np.uint8)
        rainbow = cv2.cvtColor(np.array([[rainbow]]), cv2.COLOR_HSV2BGR)[0, 0]
        # Convert headcolor to tuple of integers
        rainbow = tuple(map(int, rainbow))

        # Draw Snake
        # TODO : 아이템 먹으면 무지개 색으로 변하게?
        pts = np.array(points, np.int32)
        if len(pts.shape) == 3:
            pts = pts[:, 1]
        pts = pts.reshape((-1, 1, 2))

        # Single Mod Collision Padding Design
        if bot_flag and isMe:
            cv2.polylines(imgMain, np.int32([pts]), False, red, 25)
            cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)
        else:
            cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)

        if isMe and HandPoints and self.line_flag:
            for p in np.linspace(self.previousHead, HandPoints, 10):
                cv2.circle(imgMain, tuple(np.int32(p)), 5, (0, 255, 0), -1)

        if points:
            cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
            cv2.circle(imgMain, points[-1][1], 15, rainbow, cv2.FILLED)

        return imgMain

    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        self.socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain

    ############################################################
    def create_maze_image(self):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # White Wall
        img[np.where(self.maze_map == 1)] = (255, 255, 255)
        # Start Green
        img[np.where(self.maze_map == 2)] = (0, 255, 0)
        # End Red
        img[np.where(self.maze_map == 3)] = (0, 0, 255)
        # mid point
        # img[np.where(self.maze_map == 4)] = (255, 0, 0)
        return img

    # 내 뱀 상황 업데이트 - main에서
    def my_snake_update_menu(self, HandPoints):
        px, py = self.previousHead
        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)

        self.points.append([[px, py], [cx, cy]])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()

        menu_type = 0
        # hover event emit 할 필요 TODO
        if 490 <= cx <= 790:
            if 70 <= cy <= 170:  # menu_type: 2, SINGLE PLAY
                menu_type = 2
            elif 310 <= cy <= 410:  # menu_type: 1, MULTI PLAY
                menu_type = 1
            elif 550 <= cy <= 650:  # menu_type: 3, MAZE RUNNER
                menu_type = 3

        return menu_type

    # 내 뱀 상황 업데이트 - maze play에서
    def my_snake_update_mazeVer(self, HandPoints):
        px, py = self.previousHead
        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)

        self.points.append([[px, py], [cx, cy]])
        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()
        if self.maze_collision([cx, cy], [px, py]):
            self.passStart = False
            self.passMid = False
            self.previousHead = 0, 360
            self.points = []
            self.lengths = []
            self.currentLength = 0

        # start point 시작!
        start_pt1, start_pt2 = self.maze_start
        if (start_pt1[0] <= cx <= start_pt2[0]) and (start_pt1[1] <= cy <= start_pt2[1]):
            self.passStart = True

        # 중간 point 패스!
        mid_pt1, mid_pt2 = self.maze_mid
        if (mid_pt1[0] <= cx <= mid_pt2[0]) and (mid_pt1[1] <= cy <= mid_pt2[1]):
            if self.passStart:
                self.passMid = True

        # end point 도달
        end_pt1, end_pt2 = self.maze_end
        # print(f"end point : 1-{end_pt1}, 2-{end_pt2}")
        if (end_pt1[0] <= cx <= end_pt2[0]) and (end_pt1[1] <= cy <= end_pt2[1]):
            if self.passStart and self.passMid:
                self.maze_initialize()

    # 내 뱀 상황 업데이트
    def my_snake_update(self, HandPoints, opp_bodys):
        global bot_flag
        px, py = self.previousHead

        s_speed = 30
        cx, cy = self.set_snake_speed(HandPoints, s_speed)
        self.socketio.emit('finger_cordinate', {'head_x': cx, 'head_y': cy})

        self.points.append([[px, py], [cx, cy]])

        distance = math.hypot(cx - px, cy - py)
        self.lengths.append(distance)
        self.currentLength += distance
        self.previousHead = cx, cy

        self.length_reduction()

        if self.foodOnOff:
            self.check_snake_eating(cx, cy)

        self.send_data_to_opp()
        self.send_data_to_html()

        if self.is_udp:
            self.receive_data_from_opp()

        if opp_bodys:
            self.dist = ((self.points[-1][1][0] - opp_bodys[-1][1][0]) ** 2 + (
                    self.points[-1][1][1] - opp_bodys[-1][1][1]) ** 2) ** 0.5
        # 할일: self.multi가 false일 때, pt_dist html에 보내기
        # print(f"point distance: {pt_dist}")
        self.socketio.emit('h2h_distance', self.dist)

        opp_bodys_collsion = opp_bodys

        # Single Play Self Collision
        if bot_flag:
            opp_bodys_collsion = opp_bodys + self.points[:-3]

        if self.isCollision(self.points[-1], opp_bodys_collsion):
            global user_move
            global gameover_flag
            sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_7_path,))
            sfx_thread.start()
            gameover_flag = True
            if user_move:
                self.execute()

    def set_snake_speed(self, HandPoints, s_speed):
        px, py = self.previousHead
        # ----HandsPoint moving ----
        s_speed = 20
        if HandPoints:
            m_x, m_y = HandPoints
            dx = m_x - px  # -1~1
            dy = m_y - py

            # speed 범위: 0~1460
            if math.hypot(dx, dy) > self.maxspeed:  # 146
                self.speed = self.maxspeed
            elif math.hypot(dx, dy) < self.minspeed:
                self.speed = self.minspeed
            else:
                self.speed = math.hypot(dx, dy)

            if dx != 0:
                self.velocityX = dx / 1280
            if dy != 0:
                self.velocityY = dy / 720

            # print(self.velocityX)
            # print(self.velocityY)

        else:
            self.speed = self.minspeed

        cx = round(px + self.velocityX * self.speed)
        cy = round(py + self.velocityY * self.speed)
        # ----HandsPoint moving ----end
        if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
            if cx < 0: cx = 0
            if cx > 1280: cx = 1280
            if cy < 0: cy = 0
            if cy > 720: cy = 720

        if cx == 0 or cx == 1280:
            self.velocityX = -self.velocityX
        if cy == 0 or cy == 720:
            self.velocityY = -self.velocityY

        return cx, cy

    # 뱀 길이 조정
    def length_reduction(self):
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.lengths = self.lengths[1:]
                self.points = self.points[1:]

                if self.currentLength < self.allowedLength:
                    break

    # 뱀 식사 여부 확인
    def check_snake_eating(self, cx, cy):
        rx, ry = self.foodPoint
        if (rx - (self.wFood // 2) < cx < rx + (self.wFood // 2)) and (
                ry - (self.hFood // 2) < cy < ry + (self.hFood // 2)):
            sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_2_path,))
            sfx_thread.start()
            self.allowedLength += 50
            self.score += 1

            if self.multi:
                self.foodOnOff = False
                self.socketio.emit('user_ate_food', {'score': self.score})
            else:
                if self.score > self.bestScore:
                    self.bestScore = self.score
                    self.socketio.emit('bestScore', {'bestScore': self.bestScore})

                self.foodtimeLimit = time.time() + 11
                self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    # 뱀이 충돌했을때
    def execute(self):
        global user_number
        global user_move
        global game_over_for_debug
        # self.points = []  # all points of the snake
        # self.lengths = []  # distance between each point
        # self.currentLength = 0  # total length of the snake
        # self.allowedLength = 150  # total allowed Length
        # self.score = 0
        # self.previousHead = 0, 360
        user_move = False
        game_over_for_debug = True
        self.socketio.emit('gameover')

    def update_mazeVer(self, imgMain, HandPoints):
        self.my_snake_update_mazeVer(HandPoints)
        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)

        return imgMain

    # 송출될 프레임 업데이트
    def update(self, imgMain, HandPoints):
        global opponent_data

        opp_bodys = []
        # 0 이면 상대 뱀
        if opponent_data:
            opp_bodys = opponent_data['opp_body_node']
        imgMain = self.draw_snakes(imgMain, opp_bodys, HandPoints, 0)

        # update and draw own snake
        self.my_snake_update(HandPoints, opp_bodys)
        imgMain = self.draw_Food(imgMain)
        # 1 이면 내 뱀
        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)

        return imgMain

    # Menu 화면에서 쓰일 검은 배경 뱀
    def update_blackbg(self, imgMain, HandPoints):
        global gameover_flag, opponent_data

        # update and draw own snake
        menu_type = self.my_snake_update_menu(HandPoints)

        if self.menu_type != 0:
            if self.menu_type == menu_type:
                self.menu_time += 1

            if self.menu_time == 30:  # 5초간 menu bar에 머무른 경우
                # 할일: menu_type(1:multi, 2:single, 3:maze) 사용해서 routing
                self.socketio.emit("selected_menu_type", {'menu_type': self.menu_type})
                self.menu_time = 0
                self.menu_type = 0

        self.menu_type = menu_type

        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)

        return imgMain

    # 통신 관련 변수 설정
    def set_socket(self, my_port, opp_ip, opp_port):
        self.sock.bind(('0.0.0.0', int(my_port)))
        self.sock.settimeout(0.02)  # TODO 만약 udp, 서버 선택 오류 시 다시 0.02로
        self.opp_addr = (opp_ip, int(opp_port))

    # 데이터 전송
    def send_data_to_opp(self):
        if self.is_udp:
            data_set = str(self.points)
            self.sock.sendto(data_set.encode(), self.opp_addr)
        else:
            self.socketio.emit('game_data', {'body_node': self.points})

    def send_data_to_html(self):
        self.socketio.emit('game_data_for_debug', {'score': self.score, 'fps': self.fps})

    # 데이터 수신 (udp 통신 일때만 사용)
    def receive_data_from_opp(self):
        global opponent_data

        try:
            data, _ = self.sock.recvfrom(15000)
            decode_data = data.decode()
            if decode_data[0] == '[':
                opponent_data['opp_body_node'] = eval(decode_data)
                self.udp_count = 0
            else:
                test_code = decode_data
                self.sock.sendto(test_code.encode(), self.opp_addr)
        except socket.timeout:
            self.udp_count += 1
            if self.udp_count > 25:
                self.socketio.emit('opponent_escaped')

    # udp로 통신할지 말지
    def test_connect(self, sid):
        a = 0
        b = 0
        test_code = str(sid)

        for i in range(50):
            if i % 2 == 0:
                test_code = str(sid)
            self.sock.sendto(test_code.encode(), self.opp_addr)
            try:
                data, _ = self.sock.recvfrom(600)
                test_code = data.decode()
                if test_code == str(sid):
                    b += 1
            except socket.timeout:
                a += 1

        if a != 50 and b != 0:
            self.is_udp = False

        print(f"connection MODE : {self.is_udp} / a = {a}, b = {b}")
        self.socketio.emit('NetworkMode', {'UDP': self.is_udp})

    # 소멸자 소켓 bind 해제
    def __del__(self):
        global opponent_data
        opponent_data = {}
        self.sock.close()
    
    def bot_data_update(self):
        global bot_data, bot_cnt

        bot_speed = 10
        px, py = bot_data['bot_head_x'], bot_data['bot_head_y']

        # 1초 마다 방향 바꾸기
        # print(bot_cnt)
        if bot_cnt == 30:
            bot_data['bot_velocityX'] = random.choice([-1, 0, 1])
            if bot_data['bot_velocityX'] == 0:
                bot_data['bot_velocityY'] = random.choice([-1, 1])
            else:
                bot_data['bot_velocityY'] = random.choice([-1, 0, 1])
            bot_cnt = 0
        bot_cnt += 1

        bot_velocityX = bot_data['bot_velocityX']
        bot_velocityY = bot_data['bot_velocityY']

        cx = round(px + bot_velocityX * bot_speed)
        cy = round(py + bot_velocityY * bot_speed)

        if cx < 0 or cx > 1280 or cy < 0 or cy > 720:
            if cx < 0: cx = 0
            if cx > 1280: cx = 1280
            if cy < 0: cy = 0
            if cy > 720: cy = 720

        if cx == 0 or cx == 1280:
            bot_data['bot_velocityX'] = -bot_data['bot_velocityX']
        if cy == 0 or cy == 720:
            bot_data['bot_velocityY'] = -bot_data['bot_velocityY']

        bot_data['bot_head_x'] = cx
        bot_data['bot_head_y'] = cy
        bot_data['bot_body_node'].append([[px, py], [cx, cy]])

        distance = math.hypot(cx - px, cy - py)
        bot_data['lengths'].append(distance)
        bot_data['currentLength'] += distance

        self.socketio.emit('bot_data', {'head_x': cx, 'head_y': cy})

        if bot_data['currentLength'] > 250:
            for i, length in enumerate(bot_data['lengths']):
                bot_data['currentLength'] -= length
                bot_data['lengths'] = bot_data['lengths'][1:]
                bot_data['bot_body_node'] = bot_data['bot_body_node'][1:]

                if bot_data['currentLength'] < 250:
                    break