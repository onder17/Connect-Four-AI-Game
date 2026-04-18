import numpy as np
import random
import pygame
import sys
import math
import cv2
import json
from datetime import datetime

# Defined list for log processes
game_history_log = []
move_counter = 1

# Creating a unique file name
# Instance output: dataset_20260413_154530.json
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_dataset_file = f"datasets/dataset_{session_timestamp}.json"

playlist = [
    "sounds/connect_4_music.mpeg.ogg", 
    "sounds/point_of_maximum_force.mpeg.ogg",
    "sounds/where_the_soil_holds.mpeg.ogg",
    "sounds/a_quiet_reckoning_mpeg.ogg",
    "sounds/silence_on_peak.mpeg.ogg",
    "sounds/general_last_war.mpeg.ogg",
    "sounds/the_final_gambit.mpeg.ogg",
    "sounds/reborn.mpeg.ogg",
    "sounds/final_stage.mpeg.ogg",
    "sounds/just_tears.mpeg.ogg"
]

current_track_index = random.randint(0, len(playlist) - 1)
is_music_paused = False
MUSIC_END = pygame.USEREVENT + 1

# GAME MODES & MATCH TYPES
MODES = ["CLASSIC", "CONQUER"]
MATCH_TYPES = ["PVE", "PVP"]
current_game_mode = "CONQUER"
current_match_type = "PVE"

p1_towers = 3
p2_towers = 3

# AI DIFFICULTY LEVELS
DIFFICULTY_LEVELS = {
    "EASY": {"depth": 1, "label": "EASY"},
    "NORMAL": {"depth": 3, "label": "MEDIUM"},
    "HARD": {"depth": 5, "label": "HARD"}
}
current_difficulty = "HARD"

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

SQUARESIZE = 100
X_OFFSET = 150 # Space reserved for castles on both sides

width = COLUMN_COUNT * SQUARESIZE + (X_OFFSET * 2)
height = (ROW_COUNT+1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)

def create_board():
    board = np.zeros((ROW_COUNT,COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    
    return False

def reset_game(full_reset=True):
    global game_history_log, move_counter, current_dataset_file, p1_towers, p2_towers
    
    if full_reset:
        # Reset the list and counter because of new game
        game_history_log = []
        move_counter = 1
        p1_towers = 3
        p2_towers = 3
        
        # Creating new file name for new game
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dataset_file = f"datasets/dataset_{session_timestamp}.json"
    
    new_board = create_board()
    
    # Clear the winning message for new game
    screen.blit(game_bg_image , (0,0))
    draw_board(new_board)
    pygame.display.update()
    new_turn = random.randint(PLAYER,AI)
    return new_board , False , new_turn

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

def score_position(board, piece):
    score = 0

    # Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else: # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col

def log_final_result(winner_piece):
    if current_match_type != "PVE":
        return
        
    final_log = {
        "match_status": "GAME_OVER",
        "winner": "PLAYER" if winner_piece == PLAYER_PIECE else "AI_Warrior",
        "ai_final_status": "DEFEATED" if winner_piece == PLAYER_PIECE else "VICTORIOUS",
        "total_moves": move_counter - 1,
        "final_board": board.tolist(),
        "timestamp": str(datetime.now())
    }
    game_history_log.append(final_log)
    with open(current_dataset_file, "w", encoding="utf-8") as outfile:
        json.dump(game_history_log, outfile, indent=4, ensure_ascii=False)

def process_win(piece):
    global game_over, p1_towers, p2_towers, board, turn
    
    draw_board(board)
    screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE))
    
    # Dynamic naming based on match type
    opponent_name = "AI" if current_match_type == "PVE" else "PLAYER 2"
    
    if current_game_mode == "CONQUER":
        if explosion_sound:
            explosion_sound.play() # Explosion sound
        
        if piece == PLAYER_PIECE:
            p2_towers = max(0, p2_towers - 1)
            msg = "PLAYER 1 DESTROYS A TOWER!"
        else:
            p1_towers = max(0, p1_towers - 1)
            msg = f"{opponent_name} DESTROYS A TOWER!"
            
        label = win_font.render(msg, 1, RED if piece == PLAYER_PIECE else YELLOW)
        screen.blit(label, (width/2 - label.get_width()/2, 30))
        pygame.display.update()
        
        if p1_towers == 0 or p2_towers == 0:
            game_over = True
            if win_sound:
                win_sound.play() # Final win sound
                
            final_msg = "PLAYER 1 CONQUERS ALL!" if p1_towers > 0 else f"{opponent_name} CONQUERS ALL!"
            pygame.time.wait(1500)
            
            screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE))
            label = win_font.render(final_msg, 1, RED if p1_towers > 0 else YELLOW)
            screen.blit(label, (width/2 - label.get_width()/2, 30))
            
            log_final_result(piece)
        else:
            pygame.time.wait(2500)
            board, _, turn = reset_game(full_reset=False)
            
    else: # CLASSIC MODE
        game_over = True
        if win_sound:
            win_sound.play() # Final win sound
            
        final_msg = "PLAYER 1 WINS!!" if piece == PLAYER_PIECE else f"{opponent_name} WINS!!"
        label = win_font.render(final_msg, 1, RED if piece == PLAYER_PIECE else YELLOW)
        screen.blit(label, (width/2 - label.get_width()/2, 30))
        log_final_result(piece)

def process_draw():
    global game_over, board, turn
    
    draw_board(board)
    screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE))
    
    if current_game_mode == "CONQUER":
        msg = "BATTLE DRAW! NO TOWERS LOST!"
        label = win_font.render(msg, 1, WHITE)
        screen.blit(label, (width/2 - label.get_width()/2, 30))
        pygame.display.update()
        
        pygame.time.wait(2500)
        board, _, turn = reset_game(full_reset=False)
    else: # CLASSIC MODE
        game_over = True
        msg = "IT'S A DRAW!!"
        label = win_font.render(msg, 1, WHITE)
        screen.blit(label, (width/2 - label.get_width()/2, 30))
        pygame.display.update()

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE + X_OFFSET, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE + X_OFFSET + SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):      
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE + X_OFFSET + SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == AI_PIECE: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE + X_OFFSET + SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                
    if current_game_mode == "CONQUER":
        if castles_loaded:
            screen.blit(red_castles[p1_towers], (5, height - CASTLE_HEIGHT - 10))
            screen.blit(yellow_castles[p2_towers], (width - CASTLE_WIDTH - 5, height - CASTLE_HEIGHT - 10))
        else:
            # Fallback if images fail to load
            opponent_short = "AI" if current_match_type == "PVE" else "P2"
            p1_txt = top_button_font.render(f"P1: {p1_towers}", 1, RED)
            p2_txt = top_button_font.render(f"{opponent_short}: {p2_towers}", 1, YELLOW)
            screen.blit(p1_txt, (20, height/2))
            screen.blit(p2_txt, (width - X_OFFSET + 20, height/2))

    pygame.display.update()

def draw_button_with_hover(surface, text, font, rect_vals, color, hover_color, text_color):
    mx, my = pygame.mouse.get_pos()
    rect = pygame.Rect(rect_vals)
    if rect.collidepoint((mx, my)):
        pygame.draw.rect(surface, hover_color, rect.inflate(8, 8), border_radius=10)
    else:
        pygame.draw.rect(surface, color, rect, border_radius=10)
    rendered_text = font.render(text, 1, text_color)
    surface.blit(rendered_text, (rect.x + rect.width/2 - rendered_text.get_width()/2, rect.y + rect.height/2 - rendered_text.get_height()/2))

def play_intro_video(video_path):
    global fullscreen, screen, bg_image, game_bg_image, about_bg_image
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} file does not exist.")
        return

    # Video fps value
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    clock = pygame.time.Clock()

    # Playing voice file with pygame
    try:
        pygame.mixer.music.load("videos/intro_voice.ogg")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Video voice is not available: {e}")

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Break loop if video is over
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if fullscreen:
            info = pygame.display.Info()
            frame = cv2.resize(frame, (info.current_w, info.current_h))
        else:
            frame = cv2.resize(frame, (width, height))
            
        surf = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")
        
        screen.blit(surf, (0, 0))
        pygame.display.update()

        skip_video = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_RETURN]:
                    skip_video = True
                    break
                # F11 - fullscreen toggle
                if event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode(size, pygame.FULLSCREEN | pygame.SCALED)
                        info = pygame.display.Info()
                        bg_image = pygame.transform.smoothscale(original_bg_image, (info.current_w, info.current_h))
                        game_bg_image = pygame.transform.smoothscale(original_game_bg_image, (info.current_w, info.current_h))
                        about_bg_image = pygame.transform.smoothscale(original_about_bg_image, (info.current_w, info.current_h))
                    else:
                        screen = pygame.display.set_mode(size , pygame.SCALED)
                        bg_image = pygame.transform.smoothscale(original_bg_image, (width,height))
                        game_bg_image = pygame.transform.smoothscale(original_game_bg_image, (width,height))
                        about_bg_image = pygame.transform.smoothscale(original_about_bg_image, (width,height))
            if event.type == pygame.MOUSEBUTTONDOWN:
                skip_video = True
                break
                
        if skip_video:
            # Stop music when video is skipped
            pygame.mixer.music.stop()
            break

        clock.tick(fps)

    cap.release()
    # Stop music when intro is over
    pygame.mixer.music.stop()

pygame.init()

# Voice system initializing
pygame.mixer.init() 

# Rock falling effect
try:
    drop_sound = pygame.mixer.Sound("sounds/drop.wav")
except:
    drop_sound = None

# Explosion effect for Conquer Mode
try:
    explosion_sound = pygame.mixer.Sound("sounds/explosion_voice.ogg")
except:
    explosion_sound = drop_sound # Fallback if file is missing

# Menu click sound (requested filename)
try:
    click_sound = pygame.mixer.Sound("sounds/menu_click_voice.ogg")
except:
    click_sound = None

# Game win sound (requested filename)
try:
    win_sound = pygame.mixer.Sound("sounds/celebration_march.ogg")
except:
    win_sound = None


screen = pygame.display.set_mode(size , pygame.SCALED)

# Window title & icon settings
pygame.display.set_caption("Connect Wars: 4")
try:
    icon_image = pygame.image.load("images/pixel_sword.png")
    pygame.display.set_icon(icon_image)
except Exception as e:
    print(f"Game icon could not be loaded: {e}")

myfont = pygame.font.SysFont("monospace", 75)
button_font = pygame.font.SysFont("monospace", 30) # Small size for button
menu_font = pygame.font.SysFont("monospace", 60) # Menu Titles
win_font = pygame.font.SysFont("monospace", 50) # Font for winning text
top_button_font = pygame.font.SysFont("monospace", 22) # Elegant font for top buttons

# Game engine status
state = "MENU" # Initialize from main menu
volume_level = 0.2
fullscreen = False # Change with F11

original_bg_image = pygame.image.load("images/bg_image_new.png").convert()
bg_image = pygame.transform.smoothscale(original_bg_image, (width,height))

original_game_bg_image = pygame.image.load("images/bg_image_gameplay.jpeg").convert()
game_bg_image = pygame.transform.smoothscale(original_game_bg_image, (width,height))

original_about_bg_image = pygame.image.load("images/bg_image_about.jpeg").convert()
about_bg_image = pygame.transform.smoothscale(original_about_bg_image, (width,height))

# Load Castle Assets
try:
    CASTLE_WIDTH = 140
    CASTLE_HEIGHT = 300
    
    rc_3 = pygame.image.load("images/red_castle_initial.png").convert_alpha()
    rc_2 = pygame.image.load("images/red_castle_left_2.png").convert_alpha()
    rc_1 = pygame.image.load("images/red_castle_left_1.png").convert_alpha()
    rc_0 = pygame.image.load("images/red_castle_game_over.png").convert_alpha()
    
    yc_3 = pygame.image.load("images/yellow_castle_initial.png").convert_alpha()
    yc_2 = pygame.image.load("images/yellow_castle_left_2.png").convert_alpha()
    yc_1 = pygame.image.load("images/yellow_castle_left_1.png").convert_alpha()
    yc_0 = pygame.image.load("images/yellow_castle_game_over.png").convert_alpha()
    
    red_castles = [
        pygame.transform.smoothscale(rc_0, (CASTLE_WIDTH, CASTLE_HEIGHT)),
        pygame.transform.smoothscale(rc_1, (CASTLE_WIDTH, CASTLE_HEIGHT)),
        pygame.transform.smoothscale(rc_2, (CASTLE_WIDTH, CASTLE_HEIGHT)),
        pygame.transform.smoothscale(rc_3, (CASTLE_WIDTH, CASTLE_HEIGHT))
    ]
    
    yellow_castles = [
        pygame.transform.smoothscale(yc_0, (CASTLE_WIDTH, CASTLE_HEIGHT)),
        pygame.transform.smoothscale(yc_1, (CASTLE_WIDTH, CASTLE_HEIGHT)),
        pygame.transform.smoothscale(yc_2, (CASTLE_WIDTH, CASTLE_HEIGHT)),
        pygame.transform.smoothscale(yc_3, (CASTLE_WIDTH, CASTLE_HEIGHT))
    ]
    castles_loaded = True
except:
    castles_loaded = False
    print("Castle images missing, using fallback UI.")

# Variables initialized manually
board = create_board()
game_over = False
turn = random.randint(PLAYER, AI)

# 1. Play intro
play_intro_video("videos/intro_video.mp4")

# Obstruct the black screen while rendering main menu
pygame.event.post(pygame.event.Event(pygame.USEREVENT))

# 2. Main menu music after the intro is over
try:
    pygame.mixer.music.load(playlist[current_track_index])
    pygame.mixer.music.set_volume(volume_level)
    pygame.mixer.music.play(0)
    pygame.mixer.music.set_endevent(MUSIC_END)
except pygame.error as e:
    print(f"Main menu music did not upload: {e}")

while True: # Infinity loop structure

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == MUSIC_END:
            current_track_index = (current_track_index + 1) % len(playlist)
            pygame.mixer.music.load(playlist[current_track_index])
            pygame.mixer.music.play(0)

        # F11 & ESC function
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode(size, pygame.FULLSCREEN | pygame.SCALED)
                    info = pygame.display.Info()
                    bg_image = pygame.transform.smoothscale(original_bg_image, (info.current_w, info.current_h))
                    game_bg_image = pygame.transform.smoothscale(original_game_bg_image, (info.current_w, info.current_h))
                    about_bg_image = pygame.transform.smoothscale(original_about_bg_image, (info.current_w, info.current_h))
                else:
                    screen = pygame.display.set_mode(size , pygame.SCALED)
                    bg_image = pygame.transform.smoothscale(original_bg_image, (width,height))
                    game_bg_image = pygame.transform.smoothscale(original_game_bg_image, (width,height))
                    about_bg_image = pygame.transform.smoothscale(original_about_bg_image, (width,height))
                
                # Update and adapt to matrix belonging to screen changing
                if state == "PLAYING" or state == "PAUSED":
                    screen.blit(game_bg_image, (0,0))
                    draw_board(board)
                    pygame.display.update()
                elif state == "ABOUT":
                    screen.blit(about_bg_image, (0,0))
                    pygame.display.update()
            
            # ESC logic for Pausing
            if event.key == pygame.K_ESCAPE:
                if state == "PLAYING" and not game_over:
                    if click_sound: click_sound.play()
                    state = "PAUSED"
                elif state == "PAUSED":
                    if click_sound: click_sound.play()
                    state = "PLAYING"
                    screen.blit(game_bg_image, (0,0))
                    draw_board(board)

        # State 1 : Main menu
        if state == "MENU":
            screen.blit(bg_image , (0,0))
            
            # Fixed sizes for buttons
            draw_button_with_hover(screen, "PLAY", button_font, (width/2 - 100, 200, 200, 60), RED, (255, 100, 100), BLACK)
            draw_button_with_hover(screen, "SETTINGS", button_font, (width/2 - 100, 290, 200, 60), YELLOW, (255, 255, 150), BLACK)
            draw_button_with_hover(screen, "ABOUT", button_font, (width/2 - 100, 380, 200, 60), BLUE, (100, 100, 255), BLACK)
            draw_button_with_hover(screen, "QUIT", button_font, (width/2 - 100, 470, 200, 60), (100, 100, 100), (150, 150, 150), BLACK)
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                if (width/2 - 100 <= posx <= width/2 + 100):
                    if (200 <= posy <= 260): 
                        if click_sound: click_sound.play()
                        state = "SETUP"
                    elif (290 <= posy <= 350): 
                        if click_sound: click_sound.play()
                        state = "SETTINGS"
                    elif (380 <= posy <= 440): 
                        if click_sound: click_sound.play()
                        state = "ABOUT"
                    elif (470 <= posy <= 530): # Click quit
                        if click_sound: click_sound.play()
                        state = "QUIT_CONFIRM"

        # State 1.5 : Game Setup
        elif state == "SETUP":
            screen.blit(bg_image , (0,0))
            
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160)) 
            screen.blit(overlay, (0, 0))
            
            title_shadow = menu_font.render("GAME SETUP", 1, BLACK)
            screen.blit(title_shadow, (width/2 - title_shadow.get_width()/2 + 4, 64))
            title = menu_font.render("GAME SETUP", 1, YELLOW)
            screen.blit(title, (width/2 - title.get_width()/2, 60))
            
            # --- MODE SELECTION ---
            mode_text = button_font.render(f"MODE: {current_game_mode}", 1, WHITE)
            screen.blit(mode_text, (width/2 - mode_text.get_width()/2, 170))
            draw_button_with_hover(screen, "<", menu_font, (width/2 - 150, 220, 60, 60), YELLOW, (255, 255, 150), BLACK)
            draw_button_with_hover(screen, ">", menu_font, (width/2 + 90, 220, 60, 60), YELLOW, (255, 255, 150), BLACK)

            # --- MATCH TYPE SELECTION ---
            match_text = button_font.render(f"MATCH: {current_match_type}", 1, WHITE)
            screen.blit(match_text, (width/2 - match_text.get_width()/2, 330))
            draw_button_with_hover(screen, "<", menu_font, (width/2 - 150, 380, 60, 60), YELLOW, (255, 255, 150), BLACK)
            draw_button_with_hover(screen, ">", menu_font, (width/2 + 90, 380, 60, 60), YELLOW, (255, 255, 150), BLACK)

            # --- START & BACK BUTTONS ---
            draw_button_with_hover(screen, "START BATTLE", button_font, (width/2 - 150, 500, 300, 60), GREEN, (50, 255, 50), BLACK)
            draw_button_with_hover(screen, "BACK", button_font, (width/2 - 100, 580, 200, 60), RED, (255, 100, 100), BLACK)
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                # Mode Logic
                if (220 <= posy <= 280):
                    if (width/2 - 150 <= posx <= width/2 - 90):
                        if click_sound: click_sound.play()
                        idx = MODES.index(current_game_mode)
                        current_game_mode = MODES[(idx-1)%len(MODES)]
                    elif (width/2 + 90 <= posx <= width/2 + 150):
                        if click_sound: click_sound.play()
                        idx = MODES.index(current_game_mode)
                        current_game_mode = MODES[(idx+1)%len(MODES)]
                # Match Logic
                elif (380 <= posy <= 440):
                    if (width/2 - 150 <= posx <= width/2 - 90):
                        if click_sound: click_sound.play()
                        idx = MATCH_TYPES.index(current_match_type)
                        current_match_type = MATCH_TYPES[(idx-1)%len(MATCH_TYPES)]
                    elif (width/2 + 90 <= posx <= width/2 + 150):
                        if click_sound: click_sound.play()
                        idx = MATCH_TYPES.index(current_match_type)
                        current_match_type = MATCH_TYPES[(idx+1)%len(MATCH_TYPES)]
                # Start Battle
                elif (500 <= posy <= 560):
                    if (width/2 - 150 <= posx <= width/2 + 150):
                        if click_sound: click_sound.play()
                        board, game_over, turn = reset_game(full_reset=True)
                        state = "PLAYING"
                # Back
                elif (580 <= posy <= 640):
                    if (width/2 - 100 <= posx <= width/2 + 100):
                        if click_sound: click_sound.play()
                        state = "MENU"

        # State 2 : Settings part
        elif state == "SETTINGS":
            screen.blit(bg_image , (0,0))
            
            # Dimming overlay
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160)) # The value of 160 is transparent (0= full transparent, 255=pitch black)
            screen.blit(overlay, (0, 0))
            
            # Shaded title
            title_shadow = menu_font.render("SETTINGS", 1, BLACK)
            screen.blit(title_shadow, (width/2 - title_shadow.get_width()/2 + 4, 34))
            title = menu_font.render("SETTINGS", 1, YELLOW)
            screen.blit(title, (width/2 - title.get_width()/2, 30))
            
            # --- VOLUME SETTINGS ---
            vol_text = button_font.render(f"Music Volume: {int(volume_level*100)}%", 1, (255, 255, 255))
            screen.blit(vol_text, (width/2 - vol_text.get_width()/2, 110))
            
            draw_button_with_hover(screen, "-", menu_font, (width/2 - 100, 150, 60, 60), BLUE, (100, 100, 255), BLACK)
            draw_button_with_hover(screen, "+", menu_font, (width/2 + 40, 150, 60, 60), BLUE, (100, 100, 255), BLACK)

            # --- TRACK SETTINGS ---
            track_text = button_font.render(f"Track: {current_track_index + 1} / {len(playlist)}", 1, (255, 255, 255))
            screen.blit(track_text, (width/2 - track_text.get_width()/2, 240))

            draw_button_with_hover(screen, "<", menu_font, (width/2 - 100, 280, 60, 60), YELLOW, (255, 255, 150), BLACK)
            draw_button_with_hover(screen, ">", menu_font, (width/2 + 40, 280, 60, 60), YELLOW, (255, 255, 150), BLACK)

            # --- PAUSE BUTTON ---
            pp_color = BLACK if not is_music_paused else GREEN
            pp_hover = (50, 50, 50) if not is_music_paused else (50, 255, 50)
            pp_text_str = "PAUSE" if not is_music_paused else "CONTINUE"
            pp_text_color = WHITE if not is_music_paused else BLACK
            draw_button_with_hover(screen, pp_text_str, button_font, (width/2 - 100, 360, 200, 60), pp_color, pp_hover, pp_text_color)
            
            # --- AI LEVEL SETTINGS ---
            diff_text = button_font.render(f"AI Level: {DIFFICULTY_LEVELS[current_difficulty]['label']}", 1, (255, 255, 255))
            screen.blit(diff_text, (width/2 - diff_text.get_width()/2, 450))

            draw_button_with_hover(screen, "<", menu_font, (width/2 - 100, 490, 60, 60), YELLOW, (255, 255, 150), BLACK)
            draw_button_with_hover(screen, ">", menu_font, (width/2 + 40, 490, 60, 60), YELLOW, (255, 255, 150), BLACK)

            # --- BACK BUTTON ---
            draw_button_with_hover(screen, "BACK", button_font, (width/2 - 100, 580, 200, 60), RED, (255, 100, 100), BLACK)
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                # Volume Logic
                if (150 <= posy <= 210):
                    if (width/2 - 100 <= posx <= width/2 - 40):
                        if click_sound: click_sound.play()
                        volume_level = max(0.0, volume_level - 0.1)
                        pygame.mixer.music.set_volume(volume_level)
                    elif (width/2 + 40 <= posx <= width/2 + 100):
                        if click_sound: click_sound.play()
                        volume_level = min(1.0, volume_level + 0.1)
                        pygame.mixer.music.set_volume(volume_level)
                
                # Track Logic
                elif (280 <= posy <= 340):
                    if (width/2 - 100 <= posx <= width/2 - 40):
                        if click_sound: click_sound.play()
                        current_track_index = (current_track_index - 1) % len(playlist)
                        pygame.mixer.music.load(playlist[current_track_index])
                        pygame.mixer.music.play(0)
                        is_music_paused = False
                    elif (width/2 + 40 <= posx <= width/2 + 100):
                        if click_sound: click_sound.play()
                        current_track_index = (current_track_index + 1) % len(playlist)
                        pygame.mixer.music.load(playlist[current_track_index])
                        pygame.mixer.music.play(0)
                        is_music_paused = False
                
                # Pause Logic
                elif (360 <= posy <= 420):
                    if (width/2 - 100 <= posx <= width/2 + 100):
                        if click_sound: click_sound.play()
                        is_music_paused = not is_music_paused
                        if is_music_paused:
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()
                
                # Level Logic
                elif (490 <= posy <= 550):
                    diffs = list(DIFFICULTY_LEVELS.keys())
                    idx = diffs.index(current_difficulty)
                    if (width/2 - 100 <= posx <= width/2 - 40):
                        if click_sound: click_sound.play()
                        current_difficulty = diffs[(idx-1)%3]
                    elif (width/2 + 40 <= posx <= width/2 + 100):
                        if click_sound: click_sound.play()
                        current_difficulty = diffs[(idx+1)%3]

                # Back Logic
                elif (580 <= posy <= 640):
                    if (width/2 - 100 <= posx <= width/2 + 100):
                        if click_sound: click_sound.play()
                        state = "MENU"

        # State for ABOUT screen
        elif state == "ABOUT":
            screen.blit(about_bg_image, (0,0))
            
            # Back button - at the bottom
            draw_button_with_hover(screen, "BACK", button_font, (width/2 - 100, height - 100, 200, 60), RED, (255, 100, 100), BLACK)
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                if (width/2 - 100 <= posx <= width/2 + 100) and (height - 100 <= posy <= height - 40):
                    if click_sound: click_sound.play()
                    state = "MENU"

        # State 3 : Gameplay screen
        elif state == "PLAYING":
            # Scenario 1: Game over - wait for rematch option
            if game_over:
                # Rematch button - top right
                draw_button_with_hover(screen, "REMATCH", top_button_font, (width - 130, 30, 110, 40), (0, 200, 0), (50, 255, 50), BLACK)
                
                # Main menu button - red
                draw_button_with_hover(screen, "MENU", top_button_font, (20, 30, 120, 40), RED, (255, 100, 100), BLACK)
                
                pygame.display.update()

                # Controlization for clicking the button
                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx, posy = event.pos
                    if (width - 130 <= posx <= width - 20) and (30 <= posy <= 70):
                        if click_sound: click_sound.play()
                        board, game_over, turn = reset_game(full_reset=True) # Reset everything
                    elif (20 <= posx <= 140) and (30 <= posy <= 70):
                        if click_sound: click_sound.play()
                        state = "MENU"
                continue # Skip the loop for other playing processes

            # Scenario 2: The game is going on (classic gameplay)
            if event.type == pygame.MOUSEMOTION:
                screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                
                # Constrain visual piece within board area
                if posx < X_OFFSET: posx = X_OFFSET
                if posx > width - X_OFFSET: posx = width - X_OFFSET
                
                if turn == PLAYER:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
                elif turn == AI and current_match_type == "PVP":
                    pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
                    
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if turn == PLAYER or (turn == AI and current_match_type == "PVP"):
                    posx = event.pos[0]
                    
                    if X_OFFSET <= posx <= width - X_OFFSET:
                        screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE))
                        col = int(math.floor((posx - X_OFFSET)/SQUARESIZE))

                        if is_valid_location(board, col):
                            row = get_next_open_row(board, col)
                            active_piece = PLAYER_PIECE if turn == PLAYER else AI_PIECE
                            
                            drop_piece(board, row, col, active_piece)
                            if drop_sound:
                                drop_sound.play() # Rock falling effect

                            if winning_move(board, active_piece):
                                process_win(active_piece)
                            elif len(get_valid_locations(board)) == 0:
                                process_draw()
                            else:
                                turn += 1
                                turn = turn % 2
                            draw_board(board)

        # State 3.5 : Paused screen
        elif state == "PAUSED":
            screen.blit(game_bg_image, (0,0))
            
            for c in range(COLUMN_COUNT):
                for r in range(ROW_COUNT):
                    pygame.draw.rect(screen, BLUE, (c*SQUARESIZE + X_OFFSET, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
                    pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE + X_OFFSET + SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
            
            for c in range(COLUMN_COUNT):
                for r in range(ROW_COUNT):      
                    if board[r][c] == PLAYER_PIECE:
                        pygame.draw.circle(screen, RED, (int(c*SQUARESIZE + X_OFFSET + SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                    elif board[r][c] == AI_PIECE: 
                        pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE + X_OFFSET + SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
                        
            if current_game_mode == "CONQUER":
                if castles_loaded:
                    screen.blit(red_castles[p1_towers], (5, height - CASTLE_HEIGHT - 10))
                    screen.blit(yellow_castles[p2_towers], (width - CASTLE_WIDTH - 5, height - CASTLE_HEIGHT - 10))
                else:
                    opponent_short = "AI" if current_match_type == "PVE" else "P2"
                    p1_txt = top_button_font.render(f"P1: {p1_towers}", 1, RED)
                    p2_txt = top_button_font.render(f"{opponent_short}: {p2_towers}", 1, YELLOW)
                    screen.blit(p1_txt, (20, height/2))
                    screen.blit(p2_txt, (width - X_OFFSET + 20, height/2))

            # Dimming overlay
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160)) 
            screen.blit(overlay, (0, 0))
            
            title_shadow = menu_font.render("PAUSED", 1, BLACK)
            screen.blit(title_shadow, (width/2 - title_shadow.get_width()/2 + 4, 154))
            title = menu_font.render("PAUSED", 1, YELLOW)
            screen.blit(title, (width/2 - title.get_width()/2, 150))
            
            draw_button_with_hover(screen, "CONTINUE", button_font, (width/2 - 150, 300, 300, 60), GREEN, (50, 255, 50), BLACK)
            draw_button_with_hover(screen, "MAIN MENU", button_font, (width/2 - 150, 390, 300, 60), RED, (255, 100, 100), BLACK)
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                if (width/2 - 150 <= posx <= width/2 + 150):
                    if (300 <= posy <= 360):
                        if click_sound: click_sound.play()
                        state = "PLAYING"
                        screen.blit(game_bg_image, (0,0))
                        draw_board(board)
                    elif (390 <= posy <= 450):
                        if click_sound: click_sound.play()
                        state = "MENU"

        # State 4 : Exit confirmation screen
        elif state == "QUIT_CONFIRM":
            screen.blit(bg_image , (0,0))
            
            # Dimming overlay
            overlay = pygame.Surface((width, height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160)) # The value of 160 is transparent
            screen.blit(overlay, (0, 0))
            
            # Question text
            question = menu_font.render("QUIT THE GAME?", 1, YELLOW)
            screen.blit(question, (width/2 - question.get_width()/2, 200))
            
            # YES Button (Green)
            draw_button_with_hover(screen, "YES", button_font, (width/2 - 150, 350, 120, 60), (0, 200, 0), (50, 255, 50), BLACK)
            
            # NO Button (Red)
            draw_button_with_hover(screen, "NO", button_font, (width/2 + 30, 350, 120, 60), RED, (255, 100, 100), BLACK)
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                # YES - Exit Game
                if (width/2 - 150 <= posx <= width/2 - 30) and (350 <= posy <= 410):
                    if click_sound: click_sound.play()
                    sys.exit()
                # NO - Back to Menu
                elif (width/2 + 30 <= posx <= width/2 + 150) and (350 <= posy <= 410):
                    if click_sound: click_sound.play()
                    state = "MENU"

    # The AI movement logic
    if state == "PLAYING" and turn == AI and not game_over and current_match_type == "PVE":                
        ai_depth = DIFFICULTY_LEVELS[current_difficulty]["depth"]
        col, minimax_score = minimax(board, ai_depth, -math.inf, math.inf, True)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)
            if drop_sound:
                drop_sound.play()

            # --- EXPLAINABLE AI LOGIC (ENHANCED WITH LOSING STATES) ---
            ai_explanation = ""
            
            if minimax_score > 1000000:
                ai_explanation = "AI has found a definitive winning move or successfully blocked the opponent's absolute victory!"
                
            elif minimax_score < -1000000:
                # The AI sees it is going to lose definitively
                if current_difficulty in ["EASY", "NORMAL"]:
                    ai_explanation = f"AI ({current_difficulty}) failed to see the trap early enough. It recognizes an unavoidable defeat and makes a desperate move."
                else:
                    ai_explanation = "Despite its deep calculations, AI realizes the player has established an unblockable winning condition."
                    
            elif minimax_score <= -50:
                # The AI is in a very bad position and forced into sub-optimal defensive moves
                ai_explanation = f"AI is under heavy pressure (Score: {minimax_score}). It is struggling to defend against multiple player threats."
                
            elif minimax_score < 0:
                # Standard defensive response
                if current_difficulty == "EASY":
                    ai_explanation = f"AI (EASY) lacks deep foresight and is merely reacting to the immediate threat (Score: {minimax_score})."
                else:
                    ai_explanation = f"AI is playing defensively (Score: {minimax_score}) to neutralize the player's upcoming threats."
                    
            elif col == 3: 
                # If AI selects the exact mid column
                ai_explanation = f"AI selected the center column (Score: {minimax_score}) to maximize future horizontal and diagonal possibilities."
                
            elif minimax_score > 50: 
                # An attacking opportunity for AI
                ai_explanation = f"AI detects a strong offensive opportunity (Score: {minimax_score}) and is building a strategic trap."
                
            else: 
                # Standard steady movements
                ai_explanation = f"AI calculated {ai_depth} steps ahead and selected column {col} for steady strategic positioning."

            # Terminal Printout
            print(f"\n--- [AI DECISION CENTER] Move No: {move_counter} ---")
            print(f">> Level: {current_difficulty} (Depth: {ai_depth})")
            print(f">> Selected Column: {col}")
            print(f">> Minimax Calculated Score: {minimax_score}")
            print(f">> AI Explanation: {ai_explanation}")
            print("-" * 50)
            
            # Transforming the board to list - for JSON
            board_state_list = board.tolist() 
            move_data = {
                "move_number": move_counter,
                "player": "AI_Warrior",
                "level": current_difficulty,
                "chosen_column": int(col),
                "ai_explanation": ai_explanation,
                "minimax_calculated_score": float(minimax_score),
                "timestamp": str(datetime.now()),
                "board_state": board_state_list
            }
            game_history_log.append(move_data)
            
            # Writing to JSON file
            with open(current_dataset_file, "w", encoding="utf-8") as outfile:
                json.dump(game_history_log, outfile, indent=4, ensure_ascii=False)
                
            move_counter += 1

            if winning_move(board, AI_PIECE):
                process_win(AI_PIECE)
            elif len(get_valid_locations(board)) == 0:
                process_draw()
            else:
                turn += 1
                turn = turn % 2
                
            draw_board(board)