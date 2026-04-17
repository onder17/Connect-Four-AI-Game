import numpy as np
import random
import pygame
import sys
import math
import cv2
import json
from datetime import datetime

# --- BAŞLANGIÇ AYARLARI VE GÜNLÜKLEME ---
game_history_log = []
move_counter = 1
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_dataset_file = f"datasets/dataset_{session_timestamp}.json"

# Zorluk Şablonları (Depth: Minimax derinliği)
DIFFICULTY_LEVELS = {
    "EASY": {"depth": 1, "label": "EASY"},
    "MEDIUM": {"depth": 3, "label": "MEDIUM"},
    "HARD": {"depth": 5, "label": "HARD"}
}
current_difficulty = "MEDIUM" 

playlist = [
    "sounds/connect_4_music.mpeg.ogg", "sounds/point_of_maximum_force.mpeg.ogg",
    "sounds/where_the_soil_holds.mpeg.ogg", "sounds/a_quiet_reckoning_mpeg.ogg",
    "sounds/silence_on_peak.mpeg.ogg", "sounds/general_last_war.mpeg.ogg",
    "sounds/the_final_gambit.mpeg.ogg", "sounds/reborn.mpeg.ogg",
    "sounds/final_stage.mpeg.ogg", "sounds/just_tears.mpeg.ogg"
]

current_track_index = random.randint(0, len(playlist) - 1)
is_music_paused = False
MUSIC_END = pygame.USEREVENT + 1

# Renkler ve Boyutlar
BLUE, BLACK, RED, YELLOW, GREEN, WHITE = (0,0,255), (0,0,0), (255,0,0), (255,255,0), (0, 255, 0), (255, 255, 255)
ROW_COUNT, COLUMN_COUNT = 6, 7
PLAYER, AI = 0, 1
EMPTY, PLAYER_PIECE, AI_PIECE = 0, 1, 2
WINDOW_LENGTH = 4

# --- OYUN MANTIĞI FONKSİYONLARI ---
def create_board(): return np.zeros((ROW_COUNT,COLUMN_COUNT))
def drop_piece(board, row, col, piece): board[row][col] = piece
def is_valid_location(board, col): return board[ROW_COUNT-1][col] == 0
def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0: return r

def winning_move(board, piece):
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece: return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece: return True
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece: return True
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece: return True
    return False

def reset_game():
    global game_history_log, move_counter, current_dataset_file
    game_history_log, move_counter = [], 1
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dataset_file = f"datasets/dataset_{session_timestamp}.json"
    new_board = create_board()
    screen.blit(game_bg_image, (0,0))
    draw_board(new_board)
    return new_board, False, random.randint(PLAYER, AI)

# --- YAPAY ZEKA (MINIMAX + ALPHA-BETA) ---
def evaluate_window(window, piece):
    score, opp_piece = 0, PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    if window.count(piece) == 4: score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1: score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2: score += 2
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1: score -= 4
    return score

def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    score += center_array.count(piece) * 3
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            score += evaluate_window(row_array[c:c+WINDOW_LENGTH], piece)
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            score += evaluate_window(col_array[r:r+WINDOW_LENGTH], piece)
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            score += evaluate_window([board[r+i][c+i] for i in range(WINDOW_LENGTH)], piece)
            score += evaluate_window([board[r+3-i][c+i] for i in range(WINDOW_LENGTH)], piece)
    return score

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
    is_terminal = winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_locations) == 0
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE): return (None, 1000000000)
            if winning_move(board, PLAYER_PIECE): return (None, -1000000000)
            return (None, 0)
        return (None, score_position(board, AI_PIECE))
    
    if maximizingPlayer:
        value, column = -math.inf, random.choice(valid_locations)
        for col in valid_locations:
            b_copy = board.copy()
            drop_piece(b_copy, get_next_open_row(b_copy, col), col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value: value, column = new_score, col
            alpha = max(alpha, value)
            if alpha >= beta: break
        return column, value
    else:
        value, column = math.inf, random.choice(valid_locations)
        for col in valid_locations:
            b_copy = board.copy()
            drop_piece(b_copy, get_next_open_row(b_copy, col), col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value: value, column = new_score, col
            beta = min(beta, value)
            if alpha >= beta: break
        return column, value

# --- GÖRSEL VE ARAYÜZ FONKSİYONLARI ---
def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):      
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == AI_PIECE: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()

def draw_button_with_hover(surface, text, font, rect_vals, color, hover_color, text_color):
    mx, my = pygame.mouse.get_pos()
    rect = pygame.Rect(rect_vals)
    pygame.draw.rect(surface, hover_color if rect.collidepoint((mx, my)) else color, rect, border_radius=10)
    txt = font.render(text, 1, text_color)
    surface.blit(txt, (rect.x + rect.width/2 - txt.get_width()/2, rect.y + rect.height/2 - txt.get_height()/2))

def play_intro_video(video_path):
    global fullscreen, screen, bg_image, game_bg_image, about_bg_image
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    fps, clock = cap.get(cv2.CAP_PROP_FPS) or 30, pygame.time.Clock()
    try:
        pygame.mixer.music.load("videos/intro_voice.ogg")
        pygame.mixer.music.play()
    except: pass
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(cv2.resize(frame, (pygame.display.Info().current_w, pygame.display.Info().current_h) if fullscreen else (width, height)), cv2.COLOR_BGR2RGB)
        screen.blit(pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB"), (0, 0))
        pygame.display.update()
        if any(event.type in [pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN] for event in pygame.event.get()): break
        clock.tick(fps)
    cap.release()
    pygame.mixer.music.stop()

# --- SİSTEM BAŞLATMA ---
pygame.init()
pygame.mixer.init()
drop_sound = pygame.mixer.Sound("sounds/drop.wav")
SQUARESIZE = 100
width, height = COLUMN_COUNT * SQUARESIZE, (ROW_COUNT+1) * SQUARESIZE
size, RADIUS = (width, height), int(SQUARESIZE/2 - 5)
screen = pygame.display.set_mode(size, pygame.SCALED)
pygame.display.set_caption("Connect Wars: 4")

button_font = pygame.font.SysFont("monospace", 30)
menu_font = pygame.font.SysFont("monospace", 60)
win_font = pygame.font.SysFont("monospace", 50)
top_button_font = pygame.font.SysFont("monospace", 22)

state, volume_level, fullscreen = "MENU", 0.2, False
original_bg_image = pygame.image.load("images/bg_image_new.png").convert()
original_game_bg_image = pygame.image.load("images/bg_image_gameplay.jpeg").convert()
original_about_bg_image = pygame.image.load("images/bg_image_about.jpeg").convert()
bg_image = pygame.transform.smoothscale(original_bg_image, (width,height))
game_bg_image = pygame.transform.smoothscale(original_game_bg_image, (width,height))
about_bg_image = pygame.transform.smoothscale(original_about_bg_image, (width,height))

board, game_over, turn = reset_game()
play_intro_video("videos/intro_video.mp4")

try:
    pygame.mixer.music.load(playlist[current_track_index])
    pygame.mixer.music.set_volume(volume_level)
    pygame.mixer.music.play(0)
    pygame.mixer.music.set_endevent(MUSIC_END)
except: pass

# --- ANA DÖNGÜ ---
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
        if event.type == MUSIC_END:
            current_track_index = (current_track_index + 1) % len(playlist)
            pygame.mixer.music.load(playlist[current_track_index]); pygame.mixer.music.play(0)

        if state == "MENU":
            screen.blit(bg_image, (0,0))
            draw_button_with_hover(screen, "PLAY", button_font, (width/2-100, 200, 200, 60), RED, (255,100,100), BLACK)
            draw_button_with_hover(screen, "SETTINGS", button_font, (width/2-100, 290, 200, 60), YELLOW, (255,255,150), BLACK)
            draw_button_with_hover(screen, "ABOUT", button_font, (width/2-100, 380, 200, 60), BLUE, (100,100,255), BLACK)
            draw_button_with_hover(screen, "QUIT", button_font, (width/2-100, 470, 200, 60), (100,100,100), (150,150,150), BLACK)
            pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if width/2-100 <= x <= width/2+100:
                    if 200 <= y <= 260: board, game_over, turn = reset_game(); state = "PLAYING"
                    elif 290 <= y <= 350: state = "SETTINGS"
                    elif 380 <= y <= 440: state = "ABOUT"
                    elif 470 <= y <= 530: state = "QUIT_CONFIRM"

        elif state == "SETTINGS":
            screen.blit(bg_image, (0,0))
            ov = pygame.Surface((width, height), pygame.SRCALPHA); ov.fill((0,0,0,160)); screen.blit(ov, (0,0))
            t = menu_font.render("SETTINGS", 1, YELLOW); screen.blit(t, (width/2-t.get_width()/2, 50))
            
            # Ses Ayarı
            v_t = button_font.render(f"Volume: {int(volume_level*100)}%", 1, WHITE); screen.blit(v_t, (width/2-v_t.get_width()/2, 150))
            draw_button_with_hover(screen, "-", button_font, (width/2-100, 190, 60, 50), BLUE, (100,100,255), BLACK)
            draw_button_with_hover(screen, "+", button_font, (width/2+40, 190, 60, 50), BLUE, (100,100,255), BLACK)

            # Zorluk Ayarı (YENİ ÖZELLİK)
            d_t = button_font.render(f"Level: {DIFFICULTY_LEVELS[current_difficulty]['label']}", 1, WHITE); screen.blit(d_t, (width/2-d_t.get_width()/2, 280))
            draw_button_with_hover(screen, "<", button_font, (width/2-100, 320, 60, 50), YELLOW, (255,255,150), BLACK)
            draw_button_with_hover(screen, ">", button_font, (width/2+40, 320, 60, 50), YELLOW, (255,255,150), BLACK)

            draw_button_with_hover(screen, "BACK", button_font, (width/2-100, 500, 200, 60), RED, (255,100,100), BLACK)
            pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if 190 <= y <= 240:
                    if width/2-100 <= x <= width/2-40: volume_level = max(0.0, volume_level-0.1); pygame.mixer.music.set_volume(volume_level)
                    elif width/2+40 <= x <= width/2+100: volume_level = min(1.0, volume_level+0.1); pygame.mixer.music.set_volume(volume_level)
                elif 320 <= y <= 370:
                    diffs = list(DIFFICULTY_LEVELS.keys())
                    idx = diffs.index(current_difficulty)
                    if width/2-100 <= x <= width/2-40: current_difficulty = diffs[(idx-1)%3]
                    elif width/2+40 <= x <= width/2+100: current_difficulty = diffs[(idx+1)%3]
                elif 500 <= y <= 560: state = "MENU"

        elif state == "PLAYING":
            if game_over:
                draw_button_with_hover(screen, "REMATCH", top_button_font, (width-130, 30, 110, 40), GREEN, (50,255,50), BLACK)
                draw_button_with_hover(screen, "MENU", top_button_font, (20, 30, 120, 40), RED, (255,100,100), BLACK)
                pygame.display.update()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if width-130 <= x <= width-20: board, game_over, turn = reset_game()
                    elif 20 <= x <= 140: state = "MENU"
                continue
            if event.type == pygame.MOUSEMOTION and turn == PLAYER:
                screen.blit(game_bg_image, (0,0), (0,0,width,SQUARESIZE))
                pygame.draw.circle(screen, RED, (event.pos[0], int(SQUARESIZE/2)), RADIUS)
            pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN and turn == PLAYER:
                col = event.pos[0]//SQUARESIZE
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col); drop_piece(board, row, col, PLAYER_PIECE); drop_sound.play()
                    if winning_move(board, PLAYER_PIECE):
                        lbl = win_font.render("Player 1 wins!!", 1, RED); screen.blit(lbl, (width/2-lbl.get_width()/2, 30)); game_over = True
                    turn = (turn + 1) % 2; draw_board(board)

        elif state == "ABOUT":
            screen.blit(about_bg_image, (0,0))
            draw_button_with_hover(screen, "BACK", button_font, (width/2-100, height-100, 200, 60), RED, (255,100,100), BLACK)
            pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN: state = "MENU"

        elif state == "QUIT_CONFIRM":
            screen.blit(bg_image, (0,0))
            q = menu_font.render("QUIT?", 1, YELLOW); screen.blit(q, (width/2-q.get_width()/2, 200))
            draw_button_with_hover(screen, "YES", button_font, (width/2-150, 350, 120, 60), GREEN, (50,255,50), BLACK)
            draw_button_with_hover(screen, "NO", button_font, (width/2+30, 350, 120, 60), RED, (255,100,100), BLACK)
            pygame.display.update()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if width/2-150 <= x <= width/2-30: sys.exit()
                elif width/2+30 <= x <= width/2+150: state = "MENU"

    # --- AI HAMLE SIRASI ---
    if state == "PLAYING" and turn == AI and not game_over:
        depth = DIFFICULTY_LEVELS[current_difficulty]["depth"]
        col, score = minimax(board, depth, -math.inf, math.inf, True)
        if is_valid_location(board, col):
            row = get_next_open_row(board, col); drop_piece(board, row, col, AI_PIECE); drop_sound.play()
            
            # XAI ve JSON Loglama
            explanation = f"AI ({current_difficulty}) calculated {depth} steps and chose col {col}. Score: {score}"
            move_data = {"move": move_counter, "level": current_difficulty, "col": int(col), "score": float(score), "board": board.tolist()}
            game_history_log.append(move_data)
            with open(current_dataset_file, "w", encoding="utf-8") as f: json.dump(game_history_log, f, indent=4)
            
            print(f"--- AI DECISION --- \n{explanation}\n------------------")
            move_counter += 1
            if winning_move(board, AI_PIECE):
                lbl = win_font.render("AI wins!!", 1, YELLOW); screen.blit(lbl, (width/2-lbl.get_width()/2, 30)); game_over = True
            draw_board(board); turn = (turn + 1) % 2