import math, random, time
import cvzone
import cv2
import socket
import threading

from src.bgm.bgm import *
import numpy as np

# Color templates
red = (0, 0, 255)  # red
megenta = (255, 0, 255)  # magenta
green = (0, 255, 0)  # green
yellow = (0, 255, 255)  # yellow
cyan = (255, 255, 0)  # cyan

class MultiGameClass:
    # 생성자, class를 선언하면서 기본 변수들을 설정함
    def __init__(self,socketio, pathFood):
        self.socketio=socketio
        
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0
        self.score = 0

        self.speed = 5
        self.minspeed = 10
        self.maxspeed = math.hypot(1280, 720) / 10
        self.velocityX = random.choice([-1, 0, 1])
        self.velocityY = random.choice([-1, 1])

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 640, 360

        self.opp_score = 0
        self.opp_points = []
        self.dist = 500
        self.cut_idx = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.opp_addr = ()
        self.udp_count = 0
        self.user_number = 0
        self.queue = []

        self.is_udp = False
        self.foodOnOff = True
        self.user_move = False
        self.check_collision = False
        self.gen = True
        self.skill_flag = False
        self.opp_skill_flag = False
        self.line_flag = False

    # 통신 관련 변수 설정
    def set_socket(self, my_port, opp_ip, opp_port):
        self.sock.bind(('0.0.0.0', int(my_port)))
        self.sock.settimeout(0.02)  # TODO 만약 udp, 서버 선택 오류 시 다시 0.02로
        self.opp_addr = (opp_ip, int(opp_port))

    # udp로 통신할지 말지
    def test_connect(self, sid):
        missing_cnt = 0
        self_sid_cnt = 0
        test_code = str(sid)

        if test_code == "1":
            for i in range(50):
                if i % 3 == 0 and self_sid_cnt == 0:
                    test_code = '1'
                self.sock.sendto(test_code.encode(), self.opp_addr)
                try:
                    data, _ = self.sock.recvfrom(100)
                    test_code = data.decode()
                    if test_code == str(sid):
                        test_code = '2'
                        self_sid_cnt += 1
                        if self_sid_cnt > 3:
                            break
                except socket.timeout:
                    missing_cnt += 1

        elif test_code == "2":
            for i in range(50):
                if i % 3 == 0 and self_sid_cnt == 0:
                    test_code = '2'
                self.sock.sendto(test_code.encode(), self.opp_addr)
                try:
                    data, _ = self.sock.recvfrom(100)
                    test_code = data.decode()
                    if test_code == str(sid):
                        test_code = '1'
                        self_sid_cnt += 1
                        if self_sid_cnt > 3:
                            break
                except socket.timeout:
                    missing_cnt += 1

        # 상대로 부터 받은 본인 Player Number 카운터가 1보다 클때 UDP 연결
        if self_sid_cnt > 1:
            self.is_udp = False
            self.sock.settimeout(0.01)
            # Flushing socket buffer
            for _ in range(50):
                self.sock.recv(0)
            self.sock.settimeout(0)

        print(f"connection MODE : {self.is_udp} / missing_cnt = {missing_cnt}, self_sid_cnt = {self_sid_cnt}")
        self.socketio.emit('NetworkMode', {'UDP': self.is_udp})
        self.socketio.emit('game_ready')

    # 송출될 프레임 업데이트
    def update(self, imgMain, HandPoints):
        self.my_snake_update(HandPoints)

        if self.is_udp:
            self.receive_data_from_opp()

        imgMain = self.draw_Food(imgMain)

        # 1 이면 내 뱀 / 0 이면 상대 뱀
        imgMain = self.draw_snakes(imgMain, self.points, HandPoints, 1)
        imgMain = self.draw_snakes(imgMain, self.opp_points, HandPoints, 0)

        self.send_data_to_opp()

        if self.check_collision and self.points:
            coll_bool = self.isCollision(self.points[-1], self.opp_points)
            if coll_bool:
                if self.skill_flag:
                    self.socketio.emit("opp_cut_idx", {"cut_idx": coll_bool})
                    self.skill_flag = False
                else:
                    sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_7_path,))
                    sfx_thread.start()
                    self.execute()

        return imgMain

    # 내 뱀 상황 업데이트
    def my_snake_update(self, HandPoints):
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

        if self.opp_points and self.points:
            self.dist = ((self.points[-1][1][0] - self.opp_points[-1][1][0]) ** 2 + (
                    self.points[-1][1][1] - self.opp_points[-1][1][1]) ** 2) ** 0.5
        # 할일: self.multi가 false일 때, pt_dist html에 보내기
        # print(f"point distance: {pt_dist}")
        self.socketio.emit('h2h_distance', self.dist)

    # 뱀 속도 설정
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

            self.foodOnOff = False
            self.socketio.emit('user_ate_food', {'score': self.score})

            if self.score % 5 == 0 and self.score != 0:
                self.skill_flag = True
                sfx_thread = threading.Thread(target=play_selected_sfx, args=(sfx_3_path,))
                sfx_thread.start()

    # 먹이 그려주기
    def draw_Food(self, imgMain):
        rx, ry = self.foodPoint
        self.socketio.emit('foodPoint', {'food_x': rx, 'food_y': ry})
        imgMain = cvzone.overlayPNG(imgMain, self.imgFood, (rx - self.wFood // 2, ry - self.hFood // 2))

        return imgMain

    # 데이터 수신 (udp 통신 일때만 사용)
    def receive_data_from_opp(self):
        for _ in range(3):
            try:
                data, _ = self.sock.recvfrom(15000)
                decode_data = data.decode()
                self.queue.append(decode_data)
                self.udp_count = 0
                
            except socket.timeout:
                self.udp_count += 1
                if self.udp_count > 25:
                    self.socketio.emit('opponent_escaped')
            except BlockingIOError:
                self.udp_count += 1
                if self.udp_count > 40:
                    self.socketio.emit('opponent_escaped')

        while True:
            if len(self.queue) == 0:
                break
            elif len(self.queue) > 8:
                self.queue.pop(0)
                temp = self.queue.pop(0)
                if temp[0] == '[':
                    self.opp_points = eval(temp)
                    break
            else:
                temp = self.queue.pop(0)
                if temp[0] == '[':
                    self.opp_points = eval(temp)
                    break
                
            
    def draw_triangle(self, point, point2, size):
        x,y=point
        x2,y2=point2
        triangle_size = size
        half_triangle_size = int(triangle_size / 2)
        
        triangle = [(0, 0 - half_triangle_size),(0 - half_triangle_size, 0 + half_triangle_size),(0 + half_triangle_size, 0 + half_triangle_size)]

        angle =  math.atan2(y2-y,x2-x) -90*math.pi/180
        r_m = [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)]
            ]
        rotated_triangle = [[int(vertex[0]*r_m[0][0]+vertex[1]*r_m[0][1]+x), int(vertex[0]*r_m[1][0]+vertex[1]*r_m[1][1]+y)] for vertex in triangle]
        triangle_pts = np.array(rotated_triangle, np.int32).reshape((-1,1,2))
        return triangle_pts
    
    # 뱀 그려주기
    def draw_snakes(self, imgMain, points, HandPoints, isMe):

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

        # --- head point와 hands point 이어주기 ---
        if isMe and HandPoints and self.line_flag:
            for p in np.linspace(self.previousHead, HandPoints, 10):
                cv2.circle(imgMain, tuple(np.int32(p)), 5, (0, 255, 0), -1)

        # --- skill flag에 따라 색 바꾸기 --- 
        skill_colored = False
        if isMe:
            skill_colored = self.skill_flag
        else:
            skill_colored = self.opp_skill_flag

        if skill_colored:
            cv2.polylines(imgMain, np.int32([pts]), False, rainbow, 15)

            triangle_pts=self.draw_triangle(points[-1][1],points[-1][0], 50)
            triangle_pts_back=self.draw_triangle(points[-1][1],points[-1][0], 35)
            # cv2.polylines(imgMain, np.int32([triangle_pts1]), False, rainbow, 15)
            # cv2.polylines(imgMain, np.int32([triangle_pts2]), False, rainbow, 15)
            cv2.fillPoly(imgMain, [triangle_pts], megenta)
            cv2.fillPoly(imgMain, [triangle_pts_back], rainbow)

        else:
            cv2.polylines(imgMain, np.int32([pts]), False, maincolor, 15)
            if points:
                cv2.circle(imgMain, points[-1][1], 20, bodercolor, cv2.FILLED)
                cv2.circle(imgMain, points[-1][1], 15, rainbow, cv2.FILLED)

        return imgMain

    # 데이터 전송
    def send_data_to_opp(self):
        if self.is_udp:
            data_set = str(self.points)
            self.sock.sendto(data_set.encode(), self.opp_addr)
        else:
            self.socketio.emit('game_data', {'body_node': self.points})

    # 세 개의 점 방향성
    def ccw(self, p, a, b):
        s = p[0] * a[1] + a[0] * b[1] + b[0] * p[1]
        s -= (p[1] * a[0] + a[1] * b[0] + b[1] * p[0])

        if s > 0:
            return 1
        elif s == 0:
            return 0
        else:
            return -1

    # 두 선분의 교차 판단
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

    # 충돌 판단
    def isCollision(self, u1_head_pt, u2_pts):
        if not u2_pts:
            return 0
        # p1_a, p1_b = np.array(u1_head_pt[0]), np.array(u1_head_pt[1]) # p1_b: head point
        p1_a, p1_b = u1_head_pt[0], u1_head_pt[1]

        for idx, u2_pt in enumerate(u2_pts):
            # p2_a, p2_b = np.array(u2_pt[0]), np.array(u2_pt[1])
            p2_a, p2_b = u2_pt[0], u2_pt[1]

            if self.segmentIntersects(p1_a, p1_b, p2_a, p2_b):
                # print(p1_a, p1_b, p2_a, p2_b)
                return idx

        return 0

    # skill 사용 시 충돌 idx 자르기
    def skill_length_reduction(self):
        for i in range(self.cut_idx):
            self.currentLength -= self.lengths[i]

        if self.currentLength < 100:
            self.allowedLength = 100
        else:
            self.allowedLength = self.currentLength

        self.lengths = self.lengths[self.cut_idx:]
        self.points = self.points[self.cut_idx:]

    # 뱀이 충돌했을때
    def execute(self):
        self.check_collision = False
        self.user_move = False
        self.gen = False
        # 상대에게 게임오버 플래그 보내기 전 슬립줘서 상대 화면에도 박은게 보이게 하는 Sleep
        # time.sleep(0.25)
        self.socketio.emit('gameover')

    # 소멸자 소켓 bind 해제
    def __del__(self):
        self.sock.close()