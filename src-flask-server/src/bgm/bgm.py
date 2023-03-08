import simpleaudio as sa
import threading
import signal

# 배경음악이나 버튼음은 자바스크립트, 게임오버나 스킬 사용 효과음은 파이썬
# Global Flag for BGM status
bgm_play_obj = None
# SETTING BGM PATH
bgm_path = './src-flask-server/static/bgm/main.wav'
sfx_1_path = './src-flask-server/static/bgm/curSelect.wav'
sfx_2_path = './src-flask-server/static/bgm/eatFood.wav'
sfx_3_path = './src-flask-server/static/bgm/skill.wav'
sfx_4_path = './src-flask-server/static/bgm/gameOver.wav'
sfx_5_path = './src-flask-server/static/bgm/gameWin.wav'
sfx_6_path = './src-flask-server/static/bgm/warning.wav'
sfx_7_path = './src-flask-server/static/bgm/dead.wav'


def play_bgm():
    global bgm_play_obj
    bgm_wave_obj = sa.WaveObject.from_wave_file(bgm_path)
    bgm_play_obj = bgm_wave_obj.play()
    bgm_play_obj.wait_done()


def stop_music_exit(signal, frame):
    global bgm_play_obj
    if bgm_play_obj is not None:
        bgm_play_obj.stop()
    exit(0)


def stop_bgm():
    global bgm_play_obj
    if bgm_play_obj is not None:
        bgm_play_obj.stop()


# Create a new thread for each sound effect selected by the user
def play_selected_sfx(track):
    sfx_wave_obj = sa.WaveObject.from_wave_file(track)
    sfx_play_obj = sfx_wave_obj.play()
    sfx_play_obj.wait_done()
