import numpy as np
import random
import pygame
import sys
import math
import cv2

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

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
            
def reset_game():
    new_board = create_board()
    #clear the winning message for new game
    screen.blit(game_bg_image , (0,0))
    draw_board(new_board)
    pygame.display.update()
    new_turn  =random.randint(PLAYER,AI)
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

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
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

def play_intro_video(video_path):
    global fullscreen, screen, bg_image, game_bg_image, about_bg_image
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: {video_path} dosyası açılamadı.")
        return

    # Video FPS değerini al
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    clock = pygame.time.Clock()

    # OpenCV sesi çalmaz, bu yüzden Pygame ile ses dosyasını oynatmayı deniyoruz
    try:
        pygame.mixer.music.load("videos/intro_voice.ogg")
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Video sesi oynatılamadı: {e}")

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Video bittiyse (ret False dönerse) döngüden çık
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
                # Video izlerken F11'e basılırsa
                if event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pygame.display.set_mode(size, pygame.FULLSCREEN | pygame.SCALED)
                        info = pygame.display.Info()
                        bg_image = pygame.transform.scale(original_bg_image, (info.current_w, info.current_h))
                        game_bg_image = pygame.transform.scale(original_game_bg_image, (info.current_w, info.current_h))
                        about_bg_image = pygame.transform.scale(original_about_bg_image, (info.current_w, info.current_h))
                    else:
                        screen = pygame.display.set_mode(size , pygame.SCALED)
                        bg_image = pygame.transform.scale(original_bg_image, (width,height))
                        game_bg_image = pygame.transform.scale(original_game_bg_image, (width,height))
                        about_bg_image = pygame.transform.scale(original_about_bg_image, (width,height))
            if event.type == pygame.MOUSEBUTTONDOWN:
                skip_video = True
                break
                
        if skip_video:
            # Kullanıcı videoyu geçerse müziği durdur
            pygame.mixer.music.stop()
            break

        clock.tick(fps)

    cap.release()
    # Intro bittiğinde intro müziğini/sesini kesin durdur
    pygame.mixer.music.stop()

pygame.init()

pygame.mixer.init() #voice system initializing

#rock falling effect
drop_sound = pygame.mixer.Sound("sounds/drop.wav")


SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size , pygame.SCALED)

myfont = pygame.font.SysFont("monospace", 75)
button_font = pygame.font.SysFont("monospace", 30) #small size for button
menu_font = pygame.font.SysFont("monospace", 60) #Menu Titles
win_font = pygame.font.SysFont("monospace", 50) #font for winning text
top_button_font = pygame.font.SysFont("monospace", 22) #kibar font for top buttons

#game engine status
state = "MENU" #initialize from main menu
volume_level = 0.2
fullscreen = False #change with F11

original_bg_image = pygame.image.load("images/bg_image_new.png").convert()
bg_image = pygame.transform.scale(original_bg_image, (width,height))

original_game_bg_image = pygame.image.load("images/bg_image_gameplay.jpeg").convert()
game_bg_image = pygame.transform.scale(original_game_bg_image, (width,height))

# About ekranı için resmi yükleme
original_about_bg_image = pygame.image.load("images/bg_image_about.jpeg").convert()
about_bg_image = pygame.transform.scale(original_about_bg_image, (width,height))

#variables initialites manuel
board = create_board()
game_over = False
turn = random.randint(PLAYER, AI)

# 1. ÖNCE İNTRO VİDEOSUNU OYNAT (Ses ve Video)
play_intro_video("videos/intro_video.mp4")

# Mouse'a dokunmaya gerek kalmadan Ana Menünün hemen render alması için sahte tetikleme olayı
pygame.event.post(pygame.event.Event(pygame.USEREVENT))

# 2. İNTRO BİTTİKTEN SONRA ANA MENÜ MÜZİĞİNİ YÜKLE VE BAŞLAT
try:
    pygame.mixer.music.load("sounds/connect_4_music.mpeg.ogg")
    pygame.mixer.music.set_volume(0.2)
    pygame.mixer.music.play(-1)
except pygame.error as e:
    print(f"Menü müziği yüklenemedi: {e}")

while True: #infinity loop structure

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        #f11 function - for fullscreen
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode(size, pygame.FULLSCREEN | pygame.SCALED)
                    info = pygame.display.Info()
                    bg_image = pygame.transform.scale(original_bg_image, (info.current_w, info.current_h))
                    game_bg_image = pygame.transform.scale(original_game_bg_image, (info.current_w, info.current_h))
                    about_bg_image = pygame.transform.scale(original_about_bg_image, (info.current_w, info.current_h))
                else:
                    screen = pygame.display.set_mode(size , pygame.SCALED)
                    bg_image = pygame.transform.scale(original_bg_image, (width,height))
                    game_bg_image = pygame.transform.scale(original_game_bg_image, (width,height))
                    about_bg_image = pygame.transform.scale(original_about_bg_image, (width,height))
                #update and adapt to matrix belong to screen changing
                if state == "PLAYING":
                    draw_board(board)
                    pygame.display.update()
                elif state == "ABOUT":
                    screen.blit(about_bg_image, (0,0))
                    pygame.display.update()

        #state 1 : main menu
        if state == "MENU":
            screen.blit(bg_image , (0,0))
            
            # Y koordinatları butonların sığması için yeniden düzenlendi
            pygame.draw.rect(screen, RED, (width/2 - 100, 200, 200, 60), border_radius=10)
            play_text = button_font.render("PLAY", 1, BLACK)
            screen.blit(play_text, (width/2 - play_text.get_width()/2, 215))
            
            pygame.draw.rect(screen, YELLOW, (width/2 - 100, 290, 200, 60), border_radius=10)
            set_text = button_font.render("SETTINGS", 1, BLACK)
            screen.blit(set_text, (width/2 - set_text.get_width()/2, 305))

            pygame.draw.rect(screen, BLUE, (width/2 - 100, 380, 200, 60), border_radius=10)
            about_text = button_font.render("ABOUT", 1, BLACK)
            screen.blit(about_text, (width/2 - about_text.get_width()/2, 395))
            
            pygame.draw.rect(screen, (100, 100, 100), (width/2 - 100, 470, 200, 60), border_radius=10)
            quit_text = button_font.render("QUIT", 1, BLACK)
            screen.blit(quit_text, (width/2 - quit_text.get_width()/2, 485))
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                if (width/2 - 100 <= posx <= width/2 + 100):
                    if (200 <= posy <= 260): 
                        board, game_over, turn = reset_game() #reset everything
                        state = "PLAYING"
                    elif (290 <= posy <= 350): 
                        state = "SETTINGS"
                    elif (380 <= posy <= 440): 
                        state = "ABOUT"
                    elif (470 <= posy <= 530): # QUIT'e basıldı
                        state = "QUIT_CONFIRM"

        #state 2 : settings part
        elif state == "SETTINGS":
            screen.blit(bg_image , (0,0))
            
            title = menu_font.render("SETTINGS", 1, YELLOW)
            screen.blit(title, (width/2 - title.get_width()/2, 100))
            
            vol_text = button_font.render(f"Music Volume: {int(volume_level*100)}%", 1, RED)
            screen.blit(vol_text, (width/2 - vol_text.get_width()/2, 250))
            
            pygame.draw.rect(screen, BLUE, (width/2 - 100, 320, 60, 60), border_radius=10)
            minus_text = menu_font.render("-", 1, BLACK)
            screen.blit(minus_text, (width/2 - 80, 320))
            
            pygame.draw.rect(screen, BLUE, (width/2 + 40, 320, 60, 60), border_radius=10)
            plus_text = menu_font.render("+", 1, BLACK)
            screen.blit(plus_text, (width/2 + 55, 320))
            
            pygame.draw.rect(screen, RED, (width/2 - 100, 450, 200, 60), border_radius=10)
            back_text = button_font.render("BACK", 1, BLACK)
            screen.blit(back_text, (width/2 - back_text.get_width()/2, 465))
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                if (width/2 - 100 <= posx <= width/2 - 40) and (320 <= posy <= 380):
                    volume_level = max(0.0, volume_level - 0.1)
                    pygame.mixer.music.set_volume(volume_level)
                elif (width/2 + 40 <= posx <= width/2 + 100) and (320 <= posy <= 380):
                    volume_level = min(1.0, volume_level + 0.1)
                    pygame.mixer.music.set_volume(volume_level)
                elif (width/2 - 100 <= posx <= width/2 + 100) and (450 <= posy <= 510):
                    state = "MENU"

        #state for ABOUT screen
        elif state == "ABOUT":
            screen.blit(about_bg_image, (0,0))
            
            # BACK butonu (En alt kısma yerleştirildi)
            pygame.draw.rect(screen, RED, (width/2 - 100, height - 100, 200, 60), border_radius=10)
            back_text = button_font.render("BACK", 1, BLACK)
            screen.blit(back_text, (width/2 - back_text.get_width()/2, height - 85))
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                if (width/2 - 100 <= posx <= width/2 + 100) and (height - 100 <= posy <= height - 40):
                    state = "MENU"

        #state 3 : gameplay screen
        elif state == "PLAYING":
            #scenario 1: game over - wait for rematch option
            if game_over:
                #rematch button - top_right
                pygame.draw.rect(screen, (0, 200, 0), (width - 130, 30, 110, 40), border_radius=10)
                text = top_button_font.render("REMATCH", 1, BLACK)
                screen.blit(text, (width - 120, 40))
                
                # MAIN MENU Butonu (Sol Üst - KIRMIZI)
                pygame.draw.rect(screen, RED, (20, 30, 120, 40), border_radius=10)
                m_text = top_button_font.render("MENU", 1, BLACK)
                screen.blit(m_text, (20 + (60 - m_text.get_width()/2), 40))
                
                pygame.display.update()

                #controlization for clicking the button
                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx, posy = event.pos
                    if (width - 130 <= posx <= width - 20) and (30 <= posy <= 70):
                        board, game_over, turn = reset_game() #reset everything
                    elif (20 <= posx <= 140) and (30 <= posy <= 70):
                        state = "MENU"
                continue #skip the loop for other playing processes

            #scenario 2: the game is going on(classic gameplay)
            if event.type == pygame.MOUSEMOTION:
                screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE))
                posx = event.pos[0]
                if turn == PLAYER:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE))
                if turn == PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx/SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, PLAYER_PIECE)
                        drop_sound.play() #rock falling effect

                        if winning_move(board, PLAYER_PIECE):
                            screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE)) #clear the top section
                            label = win_font.render("Player 1 wins!!", 1, RED)
                            screen.blit(label, (width/2 - label.get_width()/2, 30))
                            game_over = True

                        turn += 1
                        turn = turn % 2
                        print_board(board)
                        draw_board(board)

        #state 4 : exit confirmation screen
        elif state == "QUIT_CONFIRM":
            screen.blit(bg_image , (0,0))
            
            # Question text
            question = menu_font.render("QUIT THE GAME?", 1, YELLOW)
            screen.blit(question, (width/2 - question.get_width()/2, 200))
            
            # YES Button (Green)
            pygame.draw.rect(screen, (0,200,0), (width/2 - 150, 350, 120, 60), border_radius=10)
            yes_text = button_font.render("YES", 1, BLACK)
            screen.blit(yes_text, (width/2 - 150 + (60 - yes_text.get_width()/2), 365))
            
            # NO Button (Red)
            pygame.draw.rect(screen, RED, (width/2 + 30, 350, 120, 60), border_radius=10)
            no_text = button_font.render("NO", 1, BLACK)
            screen.blit(no_text, (width/2 + 30 + (60 - no_text.get_width()/2), 365))
            
            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx, posy = event.pos
                # YES - Exit Game
                if (width/2 - 150 <= posx <= width/2 - 30) and (350 <= posy <= 410):
                    sys.exit()
                # NO - Back to Menu
                elif (width/2 + 30 <= posx <= width/2 + 150) and (350 <= posy <= 410):
                    state = "MENU"

    #the moving structure for ai
    if state == "PLAYING" and turn == AI and not game_over:                
        col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)
            drop_sound.play()

            if winning_move(board, AI_PIECE):
                screen.blit(game_bg_image, (0,0), (0, 0, width, SQUARESIZE)) #clear the top section
                label = win_font.render("AI wins!!", 1, YELLOW)
                screen.blit(label, (width/2 - label.get_width()/2, 30))
                game_over = True

            print_board(board)
            draw_board(board)
            turn += 1
            turn = turn % 2