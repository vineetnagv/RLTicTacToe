#implementing RL in tictactoe

import random
import pickle
import time
import tkinter as tk
from tkinter import messagebox

# func to init the game board
def initialize_board():
    """ret an empty 3x3 board."""
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# func to make a move on the board
def make_move(board, row, col, player):
    """place player's mark on board if valid."""
    if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == 0:
        board[row][col] = player
        return True
    return False

# func to check for a win
def check_win(board, player):
    """check if player has won."""
    # check rows, cols, and diags
    for r in range(3):
        if all(board[r][c] == player for c in range(3)): return True
    for c in range(3):
        if all(board[r][c] == player for r in range(3)): return True
    if all(board[i][i] == player for i in range(3)): return True
    if all(board[i][2 - i] == player for i in range(3)): return True
    return False

# func to check if board is full
def is_board_full(board):
    """check if board is full (draw)."""
    return all(board[r][c] != 0 for r in range(3) for c in range(3))

# func to check if match is over
def is_match_over(board):
    """check if game has ended."""
    return check_win(board, 1) or check_win(board, -1) or is_board_full(board)

# func to get reward
def get_reward(board):
    """get reward based on game outcome."""
    if check_win(board, 1): return 1 # agent wins
    if check_win(board, -1): return -1 # opponent wins
    return 0 # draw

# func to get available moves
def get_available_moves(board):
    """ret list of empty cells."""
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == 0]

# RL agent class
class RLAgent:
    def __init__(self, player, epsilon=1.0, min_epsilon=0.1, alpha=0.3, gamma=0.9):
        self.player = player # agent's player id (1 for X)
        self.epsilon = epsilon # exploration rate
        self.max_epsilon = epsilon  # store init val
        self.min_epsilon = min_epsilon # min exploration rate
        self.epsilon_decay = 1.0  # default to 1 (no decay) until calculated
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.states_value = {}  # dict for state values (policy)

    def get_state_value(self, state):
        """get val of a state from policy."""
        return self.states_value.get(state, 0.0)

    def choose_action(self, board):
        """choose action using epsilon-greedy policy."""
        available_moves = get_available_moves(board)
        if not available_moves:
            return None
        # exploration vs. exploitation
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves) # explore
        
        # exploit
        best_move = None
        best_value = -float('inf')
        best_moves = []
        for move in available_moves:
            next_board = [row[:] for row in board]
            make_move(next_board, move[0], move[1], self.player)
            next_state = tuple(map(tuple, next_board))
            value = self.get_state_value(next_state)
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
        
        if best_moves:
            return random.choice(best_moves)
        else:
            return random.choice(available_moves)

    def update_state_values(self, states, reward):
        """update state values using TD learning."""
        # iterate backwards thru states
        for state in reversed(states):
            value = self.get_state_value(state)
            # TD update rule
            value += self.alpha * (self.gamma * reward - value)
            self.states_value[state] = value
            reward = value # update reward for prev state

    def calculate_decay(self, total_episodes):
        """calc exponential decay rate for epsilon."""
        if total_episodes > 0:
            # formula: decay_rate = (end_val / start_val)^(1/num_steps)
            self.epsilon_decay = (self.min_epsilon / self.max_epsilon) ** (1 / total_episodes)
        else:
            self.epsilon_decay = 1.0

    def decay_epsilon(self):
        """apply decay to epsilon."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# func to train the RL agent
def train(agent, opponent_player, episodes=4201337):
    """train agent via self-play."""
    # calc decay rate based on total episodes
    agent.calculate_decay(episodes)

    # points to save policy at
    save_points = {
        int(episodes * 0.01): "1%",
        int(episodes * 0.05): "5%",
        int(episodes * 0.10): "10%",
        int(episodes * 0.20): "20%",
        int(episodes * 0.50): "50%",
        int(episodes * 0.75): "75%",
        episodes: "100%",
    }
    for episode in range(episodes):
        board = initialize_board()
        current_player = random.choice([1, -1])
        agent_states = []

        while not is_match_over(board):
            if current_player == agent.player:
                action = agent.choose_action(board)
                if action:
                    make_move(board, action[0], action[1], agent.player)
                    agent_states.append(tuple(map(tuple, board)))
            else: # opponent's turn
                if not is_match_over(board):
                    moves = get_available_moves(board)
                    if moves:
                        move = random.choice(moves) # random opponent
                        make_move(board, move[0], move[1], opponent_player)
            
            current_player *= -1 # switch player
        
        reward = get_reward(board)
        agent.update_state_values(agent_states, reward)
        
        # apply decay at end of episode
        agent.decay_epsilon()

        # save policy at checkpoints
        if (episode + 1) in save_points:
            percentage = save_points[episode + 1]
            filename = f"rl_agent_policy_at_{percentage}.pkl"
            save_policy(agent, filename)
            print(f"Policy saved to {filename} at {percentage} completion ({episode + 1} episodes).")

# func to save agent's policy
def save_policy(agent, filename="rl_agent_policy.pkl"):
    """save agent's states_value dict to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(agent.states_value, f)
    print(f"Policy saved to {filename}")

# func to load agent's policy
def load_policy(agent):
    """load policy from file, trying latest first."""
    policy_files = [
        "rl_agent_policy_at_100%.pkl",
        "rl_agent_policy_at_75%.pkl",
        "rl_agent_policy_at_50%.pkl",
        "rl_agent_policy_at_20%.pkl",
        "rl_agent_policy_at_10%.pkl",
        "rl_agent_policy_at_5%.pkl",
        "rl_agent_policy_at_1%.pkl"
    ]
    for filename in policy_files:
        try:
            with open(filename, 'rb') as f:
                agent.states_value = pickle.load(f)
            print(f"Policy loaded from {filename}")
            return True
        except FileNotFoundError:
            continue
    print("No saved policy found. Starting with a new one.")
    return False

# GUI class for Tic-Tac-Toe
class TicTacToeGUI:
    def __init__(self, root, agent):
        self.root = root
        self.root.title("Tic-Tac-Toe")
        self.agent = agent
        self.board = initialize_board()
        self.current_player = 1 # default starting player
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.game_over = False
        self.start_time = None
        self.status_label = None
        self.choice_frame = None
        self.create_widgets(initial_state='disabled')
        self.setup_choice_screen()

    def setup_choice_screen(self):
        """creates screen for player to choose who goes first."""
        self.choice_frame = tk.Frame(self.root, bg='grey')
        self.choice_frame.place(relwidth=1, relheight=1)

        content_frame = tk.Frame(self.choice_frame, bg='grey')
        content_frame.pack(expand=True)

        label = tk.Label(content_frame, text="choose.", font=('normal', 16), bg='grey')
        label.pack(pady=10)

        button_frame = tk.Frame(content_frame, bg='grey')
        button_frame.pack(pady=10)

        agent_button = tk.Button(button_frame, text="i want to not win.", command=lambda: self.start_game(1), fg="red")
        agent_button.pack(side=tk.LEFT, padx=5)

        player_button = tk.Button(button_frame, text="i want to go first.", command=lambda: self.start_game(-1), fg="blue")
        player_button.pack(side=tk.RIGHT, padx=5)

        watermark = tk.Label(self.choice_frame, text="made by vinz", font=('normal', 8), bg='grey')
        watermark.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

    def start_game(self, starting_player):
        """starts the game with the selected player."""
        self.current_player = starting_player
        self.choice_frame.destroy()
        self.enable_buttons()
        if self.current_player == self.agent.player:
            self.root.after(500, self.agent_move)

    def create_widgets(self, initial_state='normal'):
        """create GUI buttons."""
        self.status_label = tk.Label(self.root, text="Time Elapsed: 0s", font=('normal', 12))
        self.status_label.grid(row=3, columnspan=3)
        for r in range(3):
            for c in range(3):
                self.buttons[r][c] = tk.Button(self.root, text='', font=('normal', 40), width=5, height=2,
                                               command=lambda r=r, c=c: self.on_button_click(r, c),
                                               state=initial_state, bg='lightgrey' if initial_state == 'disabled' else 'white')
                self.buttons[r][c].grid(row=r, column=c)

    def enable_buttons(self):
        """enables all buttons on the grid."""
        for r in range(3):
            for c in range(3):
                self.buttons[r][c].config(state='normal', bg='white')

    def update_time(self):
        """update elapsed time label."""
        if self.game_over:
            return
        if self.start_time:
            elapsed_time = int(time.time() - self.start_time)
            self.status_label.config(text=f"Time Elapsed: {elapsed_time}s")
        self.root.after(1000, self.update_time)

    def on_button_click(self, r, c):
        """handle human player's move."""
        if not self.start_time:
            self.start_time = time.time()
            self.update_time()
        # check if move is valid
        if self.board[r][c] == 0 and self.current_player == -1 and not self.game_over:
            make_move(self.board, r, c, -1)
            self.update_board()
            if self.check_game_over():
                return
            self.current_player = 1
            self.root.after(500, self.agent_move) # agent's turn

    def agent_move(self):
        """handle agent's move."""
        if not self.start_time:
            self.start_time = time.time()
            self.update_time()
        if self.current_player == self.agent.player and not self.game_over:
            action = self.agent.choose_action(self.board)
            if action:
                make_move(self.board, action[0], action[1], self.agent.player)
                self.update_board()
                if self.check_game_over():
                    return
                self.current_player = -1 # human's turn

    def update_board(self):
        """update GUI board display."""
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 1:
                    self.buttons[r][c].config(text='X', state='disabled', disabledforeground='red')
                elif self.board[r][c] == -1:
                    self.buttons[r][c].config(text='O', state='disabled', disabledforeground='blue')

    def check_game_over(self):
        """check for game over and display result."""
        if check_win(self.board, 1):
            self.status_label.config(text="X wins! Thanks for playing! :)")
            self.game_over = True
            return True
        elif check_win(self.board, -1):
            self.status_label.config(text="O wins! Thanks for playing! :)")
            self.game_over = True
            return True
        elif is_board_full(self.board):
            self.status_label.config(text="It's a draw! Thanks for playing! :)")
            self.game_over = True
            return True
        return False

if __name__ == "__main__":
    # create RL agent
    rl_agent = RLAgent(player=1)
    
    # load policy if exists, else train
    if not load_policy(rl_agent):
        print("Training the RL agent...")
        train(rl_agent, opponent_player=-1)
        print("Training finished.")

    # turn off exploration for gameplay
    rl_agent.epsilon = 0 
    
    # create and run GUI
    root = tk.Tk()
    gui = TicTacToeGUI(root, rl_agent)
    root.mainloop()
