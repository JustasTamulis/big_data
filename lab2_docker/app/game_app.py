import streamlit as st
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt
from PIL import Image
import streamlit.components.v1 as components

# --- Game Configuration ---
INITIAL_MAP_SIZE = 10
INITIAL_PLAYER_MOVES = 2
INITIAL_ENEMY_MOVES = 3
PLAYER_CHAR = '$'
ENEMY_CHAR = 'T'
WALL_CHAR = '@'
EMPTY_CHAR = '.'
DANGER_CHAR = '+'  # New character for danger zones
HIGH_SCORE_FILE = "bestRez_streamlit.txt"
IMAGE_DIR = "img"
IMG_SIZE_PX = 60  # Increased for bigger icons

# CSS for more consistent grid spacing
GRID_CSS = """
<style>
    div.row-widget.stHorizontal > div {
        padding: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
    }
    div.element-container {padding: 0 !important; margin: 0 !important;}
    .stImage img {margin: 0 !important; padding: 0 !important;}
    
    /* Hide buttons visually while keeping them in DOM for keyboard controls */
    .game-controls-container .stButton button {
        opacity: 0.1;
        transform: scale(0.5);
        margin: -5px;
        padding: 0;
        font-size: 0.6em;
        min-height: 0;
        height: 20px !important;
    }
    
    .game-controls-container .stButton button:hover {
        opacity: 0.7;
        transform: scale(0.7);
    }
</style>
"""

# --- Image Generation ---
def generate_tile_images(img_dir=IMAGE_DIR, size=IMG_SIZE_PX):
    """Generates and saves simple tile images if they don't exist."""
    os.makedirs(img_dir, exist_ok=True)
    tiles = {
        PLAYER_CHAR: ('#1E88E5', os.path.join(img_dir, 'player.png')),  # Distinct deep blue
        ENEMY_CHAR: ('#D32F2F', os.path.join(img_dir, 'enemy.png')),    # Distinct deep red
        WALL_CHAR: ('#424242', os.path.join(img_dir, 'wall.png')),      # Dark gray
        EMPTY_CHAR: ('#F5F5F5', os.path.join(img_dir, 'empty.png')),    # Light gray
        DANGER_CHAR: ('#FFCDD2', os.path.join(img_dir, 'empty_highlighted.png'))  # Light red for danger zones
    }

    # Generate new tile images with more distinct colors and larger shapes
    for char, (color, path) in tiles.items():
        # Always regenerate for updated styling
        fig, ax = plt.subplots(figsize=(1, 1))
        
        if char == PLAYER_CHAR:
            # Blue circle for player (larger)
            circle = plt.Circle((0.5, 0.5), 0.4, color=color)
            ax.add_patch(circle)
        elif char == ENEMY_CHAR:
            # Red X for enemy (larger)
            ax.plot([0.2, 0.8], [0.2, 0.8], color=color, linewidth=6)
            ax.plot([0.2, 0.8], [0.8, 0.2], color=color, linewidth=6)
        elif char == WALL_CHAR:
            # Gray solid square for wall
            rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, color=color)
            ax.add_patch(rect)
        elif char == DANGER_CHAR:
            # Light red background with pattern for danger
            rect = plt.Rectangle((0, 0), 1, 1, color=color)
            ax.add_patch(rect)
            # Add subtle pattern to indicate danger
            for i in range(0, 10, 2):
                ax.plot([i/10, (i+1)/10], [0, 1], color='#EF9A9A', alpha=0.5, linewidth=1)
        else:
            # Plain background for empty
            rect = plt.Rectangle((0, 0), 1, 1, color=color)
            ax.add_patch(rect)
            
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=size)
        plt.close(fig)
    loaded_images = {char: Image.open(path) for char, (_, path) in tiles.items()}
    return loaded_images

@st.cache_data
def load_images():
    return generate_tile_images()

# --- Danger Zone Calculation ---
def calculate_danger_zones(game_map, enemy_pos, max_distance=3):
    """Calculates areas within specified distance of the enemy"""
    rows, cols = game_map.shape
    e_r, e_c = enemy_pos
    danger_positions = set()
    
    # Use BFS to find all cells within max_distance moves
    queue = deque([(e_r, e_c, 0)])  # (row, col, distance)
    visited = set([(e_r, e_c)])
    
    while queue:
        r, c, dist = queue.popleft()
        
        if 0 < dist <= max_distance:
            danger_positions.add((r, c))
            
        if dist < max_distance:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if (0 <= nr < rows and 0 <= nc < cols and 
                    game_map[nr, nc] != WALL_CHAR and 
                    (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))
    
    return danger_positions

def apply_danger_overlay(game_map, enemy_pos):
    """Applies the danger zone overlay to a copy of the game map"""
    display_map = game_map.copy()
    danger_positions = calculate_danger_zones(game_map, enemy_pos)
    
    # Apply the danger marking to empty tiles only
    for r, c in danger_positions:
        if display_map[r, c] == EMPTY_CHAR:
            display_map[r, c] = DANGER_CHAR
            
    return display_map

# --- Helper Functions ---
def display_map(game_map, images):
    """Displays the game map using images in a more compact grid."""
    rows, cols = game_map.shape
    
    # Apply danger zone overlay if enemy is present
    enemy_pos = find_char(game_map, ENEMY_CHAR)
    if enemy_pos[0] is not None:
        display_map = apply_danger_overlay(game_map, enemy_pos)
    else:
        display_map = game_map
    
    # Add custom CSS for grid spacing
    st.markdown(GRID_CSS, unsafe_allow_html=True)
    
    # Use container with custom CSS for tight spacing
    with st.container():
        for r in range(rows):
            # Create columns with equal spacing
            row_cols = st.columns(cols, gap="small")
            for c in range(cols):
                char = display_map[r, c]
                if char in images:
                    with row_cols[c]:
                        # Use a consistent image size and remove padding
                        st.image(images[char], width=IMG_SIZE_PX, use_container_width=False)

# Function to load the JavaScript for keyboard controls
def load_js_hack():
    with open("app/index.html", "r") as f:
        js_code = f.read()
    return js_code

def find_char(game_map, char):
    """Finds the coordinates of a character on the map."""
    rows, cols = game_map.shape
    for r in range(rows):
        for c in range(cols):
            if game_map[r, c] == char:
                return r, c
    return None, None

def bfs_find_path(game_map, start_pos, target_pos):
    """
    Performs Breadth-First Search to find the next step towards the target.
    Returns the next position (row, col) for the enemy or None if no path.
    """
    rows, cols = game_map.shape
    q = deque([(start_pos, [])])
    visited = {start_pos}
    parent = {start_pos: None}

    while q:
        (r, c), path = q.popleft()

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc

            if (nr, nc) == target_pos:
                curr = (r, c)
                while parent[curr] != start_pos and parent[curr] is not None:
                    curr = parent[curr]
                if parent[curr] is None and curr == start_pos:
                    return (nr, nc)
                return curr

            if 0 <= nr < rows and 0 <= nc < cols and \
               game_map[nr, nc] != WALL_CHAR and \
               (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                new_path = path + [(nr, nc)]
                q.append(((nr, nc), new_path))

    return None

def load_high_score():
    if os.path.exists(HIGH_SCORE_FILE):
        try:
            with open(HIGH_SCORE_FILE, 'r') as f:
                return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            return 0
    return 0

def save_high_score(score):
    try:
        with open(HIGH_SCORE_FILE, 'w') as f:
            f.write(str(score))
    except IOError:
        st.warning("Could not save high score.")

# --- Game Initialization ---
def init_game():
    st.session_state.map_size = INITIAL_MAP_SIZE
    st.session_state.player_moves_per_turn = INITIAL_PLAYER_MOVES
    st.session_state.enemy_moves_per_turn = INITIAL_ENEMY_MOVES

    map_size = st.session_state.map_size
    game_map = np.full((map_size + 2, map_size + 2), EMPTY_CHAR, dtype=str)
    game_map[0, :] = WALL_CHAR
    game_map[-1, :] = WALL_CHAR
    game_map[:, 0] = WALL_CHAR
    game_map[:, -1] = WALL_CHAR

    p_r, p_c = 1, 1
    e_r, e_c = map_size, map_size

    game_map[p_r, p_c] = PLAYER_CHAR
    game_map[e_r, e_c] = ENEMY_CHAR

    st.session_state.game_map = game_map
    st.session_state.player_pos = (p_r, p_c)
    st.session_state.enemy_pos = (e_r, e_c)
    st.session_state.player_moves_left = st.session_state.player_moves_per_turn
    st.session_state.score = 0
    st.session_state.game_over = False
    st.session_state.message = "Game started. Your turn! Use keyboard."
    st.session_state.high_score = load_high_score()
    st.session_state.game_started = True
    st.session_state.images = load_images()

# --- Game Logic ---
def move_player(dr, dc):
    """Move player in the specified direction"""
    if st.session_state.game_over or st.session_state.player_moves_left <= 0:
        return

    p_r, p_c = st.session_state.player_pos
    nr, nc = p_r + dr, p_c + dc

    if st.session_state.game_map[nr, nc] == EMPTY_CHAR:
        st.session_state.game_map[p_r, p_c] = EMPTY_CHAR
        st.session_state.game_map[nr, nc] = PLAYER_CHAR
        st.session_state.player_pos = (nr, nc)
        st.session_state.player_moves_left -= 1
        st.session_state.score += 1
        st.session_state.message = "Moved."
        check_turn_end()
    elif st.session_state.game_map[nr, nc] == ENEMY_CHAR:
        st.session_state.message = f"Game Over! Enemy caught you. Final Score: {st.session_state.score}"
        st.session_state.game_over = True
        if st.session_state.score > st.session_state.high_score:
            save_high_score(st.session_state.score)
            st.session_state.high_score = st.session_state.score
    else:
        st.session_state.message = "Cannot move there (Wall)."

def place_wall(dr, dc):
    """Place a wall adjacent to the player"""
    if st.session_state.game_over or st.session_state.player_moves_left <= 0:
        return

    p_r, p_c = st.session_state.player_pos
    nr, nc = p_r + dr, p_c + dc

    map_rows, map_cols = st.session_state.game_map.shape
    if not (0 <= nr < map_rows and 0 <= nc < map_cols):
        st.session_state.message = "Cannot place wall outside map."
        return

    if st.session_state.game_map[nr, nc] == EMPTY_CHAR:
        st.session_state.game_map[nr, nc] = WALL_CHAR
        st.session_state.player_moves_left -= 1
        st.session_state.message = "Placed wall."
        check_turn_end()
    elif st.session_state.game_map[nr, nc] == WALL_CHAR:
        st.session_state.message = "There's already a wall there."
    elif st.session_state.game_map[nr, nc] == ENEMY_CHAR:
        st.session_state.message = "Cannot place wall on the enemy."
    elif st.session_state.game_map[nr, nc] == PLAYER_CHAR:
        st.session_state.message = "Cannot place wall on yourself."
    else:
        st.session_state.message = "Cannot place wall there."

def enemy_turn():
    """Enemy turn logic with visual path indicators"""
    if st.session_state.game_over:
        return

    st.session_state.message = "Enemy's turn..."
    e_r, e_c = st.session_state.enemy_pos
    p_r, p_c = st.session_state.player_pos
    
    # Calculate all enemy moves at once
    path = []
    current_pos = (e_r, e_c)
    
    for move_num in range(st.session_state.enemy_moves_per_turn):
        if st.session_state.game_over:
            break
            
        map_copy = st.session_state.game_map.copy()
        next_pos = bfs_find_path(map_copy, current_pos, (p_r, p_c))
        
        if next_pos:
            ne_r, ne_c = next_pos
            path.append((ne_r, ne_c))
            
            # Check if player will be caught
            if map_copy[ne_r, ne_c] == PLAYER_CHAR:
                st.session_state.game_over = True
                break
            
            # Update for next iteration
            current_pos = (ne_r, ne_c)
        else:
            # No path found = player wins
            st.session_state.message = f"Game Over! Enemy cannot reach you. Final Score: {st.session_state.score}"
            st.session_state.game_over = True
            if st.session_state.score > st.session_state.high_score:
                save_high_score(st.session_state.score)
                st.session_state.high_score = st.session_state.score
            break
    
    # Now apply all moves at once with a visualization of the path
    if path:
        # Apply all moves
        for i, (ne_r, ne_c) in enumerate(path):
            # Clear the previous enemy position
            if i == 0:
                st.session_state.game_map[e_r, e_c] = EMPTY_CHAR
            else:
                prev_r, prev_c = path[i-1]
                st.session_state.game_map[prev_r, prev_c] = EMPTY_CHAR
                
            # Place enemy at new position
            if st.session_state.game_map[ne_r, ne_c] == PLAYER_CHAR:
                # Player caught - game over
                st.session_state.game_map[ne_r, ne_c] = ENEMY_CHAR
                st.session_state.enemy_pos = (ne_r, ne_c)
                st.session_state.message = f"Game Over! Enemy caught you. Final Score: {st.session_state.score}"
                st.session_state.game_over = True
                if st.session_state.score > st.session_state.high_score:
                    save_high_score(st.session_state.score)
                    st.session_state.high_score = st.session_state.score
                break
            else:
                # Regular move
                st.session_state.game_map[ne_r, ne_c] = ENEMY_CHAR
                
        # Update enemy position to final position
        if path and not st.session_state.game_over:
            st.session_state.enemy_pos = path[-1]
            st.session_state.message = f"Enemy moved {len(path)} times. Your turn!"
            st.session_state.player_moves_left = st.session_state.player_moves_per_turn
    else:
        # No moves were made
        st.session_state.player_moves_left = st.session_state.player_moves_per_turn
        st.session_state.message = "Enemy couldn't move. Your turn!"

def check_turn_end():
    """Check if the player's turn should end"""
    p_r, p_c = st.session_state.player_pos
    can_move = False
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = p_r + dr, p_c + dc
        map_rows, map_cols = st.session_state.game_map.shape
        if 0 <= nr < map_rows and 0 <= nc < map_cols and st.session_state.game_map[nr, nc] == EMPTY_CHAR:
            can_move = True
            break
    if not can_move and st.session_state.player_moves_left > 0:
        can_place_wall = False
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = p_r + dr, p_c + dc
            map_rows, map_cols = st.session_state.game_map.shape
            if 0 <= nr < map_rows and 0 <= nc < map_cols and st.session_state.game_map[nr, nc] == EMPTY_CHAR:
                can_place_wall = True
                break
        if not can_place_wall:
            st.warning("Player is trapped with no valid moves or wall placements left. Ending turn.")
            st.session_state.player_moves_left = 0

    if st.session_state.player_moves_left <= 0 and not st.session_state.game_over:
        enemy_turn()

def skip_turn():
    """Skip the rest of the player's turn"""
    if not st.session_state.game_over and st.session_state.player_moves_left > 0:
        st.session_state.player_moves_left = 0
        st.session_state.message = "Turn skipped."
        check_turn_end()

def shrink():
    """Shrink the button expander"""
    st.session_state.expanded = False

if 'expanded' not in st.session_state:
    st.session_state['expanded'] = True

# --- Streamlit UI ---
st.set_page_config(layout="wide")

if 'game_started' not in st.session_state:
    init_game()

if not st.session_state.game_over:
    col1, col2 = st.columns([3, 2])  # Make game map column wider

    with col1:
        display_map(st.session_state.game_map, st.session_state.images)

    with col2:
        st.header("Status")
        score_col, high_score_col = st.columns(2)
        with score_col:
            st.write(f"**Score:** {st.session_state.score}")
        with high_score_col:    
            st.write(f"**High Score:** {st.session_state.high_score}")
        
        st.write(f"**Moves Left:** {st.session_state.player_moves_left}")
        st.info(st.session_state.message)

        # Controls (buttons for JavaScript to interact with)
        st.subheader("Controls")
        st.markdown("""
        *   **Arrow Keys:** Move Player (blue)
        *   **W, A, S, D:** Place Wall (gray) adjacent
        *   **O:** Skip remaining moves / End Turn
        """)
    
        # Add a container div with the special class for button styling
        st.markdown('<div class="game-controls-container">', unsafe_allow_html=True)
        
        # Movement buttons
        move_col1, move_col2, move_col3 = st.columns(3)
        with move_col1:
            st.write("")  # Empty space
        with move_col2:
            st.button("Up", on_click=move_player, args=(-1, 0))
        with move_col3:
            st.write("")  # Empty space
            
        move_col1, move_col2, move_col3 = st.columns(3)
        with move_col1:
            st.button("Left", on_click=move_player, args=(0, -1))
        with move_col2:
            st.button("Down", on_click=move_player, args=(1, 0))
        with move_col3:
            st.button("Right", on_click=move_player, args=(0, 1))
        
        # Wall placement buttons
        st.markdown("---")
        wall_col1, wall_col2, wall_col3 = st.columns(3)
        with wall_col1:
            st.write("")  # Empty space
        with wall_col2:
            st.button("Wall Up", on_click=place_wall, args=(-1, 0))
        with wall_col3:
            st.write("")  # Empty space
            
        wall_col1, wall_col2, wall_col3 = st.columns(3)
        with wall_col1:
            st.button("Wall Left", on_click=place_wall, args=(0, -1))
        with wall_col2:
            st.button("Wall Down", on_click=place_wall, args=(1, 0))
        with wall_col3:
            st.button("Wall Right", on_click=place_wall, args=(0, 1))

        # Skip turn button
        st.markdown("---")
        st.button("Skip Turn", on_click=skip_turn)
        
        # Close the container div
        st.markdown('</div>', unsafe_allow_html=True)

    # Load the JavaScript keyboard hack
    components.html(load_js_hack(), height=0, width=0)

else:
    # Game over screen
    col1, col2 = st.columns([3, 2])
    
    with col1:
        display_map(st.session_state.game_map, st.session_state.images)
    
    with col2:
        st.header("Game Over!")
        st.error(st.session_state.message)

        st.metric("Score", st.session_state.score)
        st.metric("High Score", st.session_state.high_score)
        
        # Restart button
        if st.button("Play Again?"):
            init_game()
            st.rerun()

# Hidden element for React to render before JavaScript runs
# This ensures that buttons are available when the keyboard script runs
st.write("", unsafe_allow_html=True)
