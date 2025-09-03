
# --- IMPORT STATEMENTS ---
# Imports are used to bring in pre-written code from other files or libraries to use in this program.
# Without these imports, you would have to write the functionality they provide from scratch.

# The 'random' library provides functions for generating random numbers and making random choices.
# It is used here for two key things:
# 1. In the RL agent's "exploration" phase, where it tries a random move to learn new strategies.
# 2. To make the opponent's moves, creating a simple, unpredictable opponent for the agent to train against.
# Without it, the agent could not explore, and the opponent would have no logic to make moves.
import random

# The 'pickle' library is used for serializing and de-serializing Python objects.
# "Serializing" means converting an object in memory (like our agent's 'brain' or policy) into a stream of bytes
# that can be saved to a file. "De-serializing" is the reverse process of loading it back into memory.
# It is used here to save the agent's training progress and load it back later, so we don't have to
# re-train the agent every time we run the program.
# Without it, the agent would start with no knowledge every time the program runs.
import pickle

# The 'time' library provides time-related functions.
# It is used in the GUI part of the code to measure and display the time elapsed during a game.
# It helps add a user-friendly feature but is not critical to the game's or agent's logic.
# Without it, the game timer feature would not work.
import time

# The 'tkinter' library is Python's standard library for creating graphical user interfaces (GUIs).
# It's used here to create the visual Tic-Tac-Toe board, buttons, and text that the user interacts with.
# The entire visual and interactive part of the application depends on tkinter.
# 'tk' is a conventional alias (a shorter name) for tkinter to make the code easier to write and read.
# Without it, the game would have to be played in the console, with no graphical elements.
import tkinter as tk

# From the tkinter library, we specifically import the 'messagebox' module.
# This module allows for the creation of standard pop-up dialog boxes (e.g., information, warning, error).
# While it's imported here, it is not actually used in the final version of this code, but it could be
# used to show the game's result in a pop-up, for example.
from tkinter import messagebox

# --- GAME LOGIC FUNCTIONS ---
# These functions define the fundamental rules and state manipulations of the Tic-Tac-Toe game.
# They are independent of the RL agent or the GUI, focusing solely on the game's mechanics.

# This function's name is 'initialize_board'. It takes no arguments.
def initialize_board():
    """ret an empty 3x3 board.""" # This is a docstring, explaining what the function does.
    
    # It returns a data structure representing the game board.
    # The data structure used is a list of lists, which effectively creates a 2D grid or matrix.
    # - The outer list `[...]` contains three elements.
    # - Each of these elements is another list `[0, 0, 0]`, representing a row.
    # In this board representation:
    # - `0` signifies an empty cell.
    # - `1` will signify a move by Player 'X' (our RL agent).
    # - `-1` will signify a move by Player 'O' (the opponent/human).
    # Using numbers is computationally efficient and common in game AI.
    # Without this function, we would have no way to start a new game with a clean board.
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# This function is named 'make_move' and takes four arguments: the board, row, col, and player.
def make_move(board, row, col, player):
    """place player's mark on board if valid."""
    
    # This `if` statement checks if a move is valid. It has three conditions:
    # 1. `0 <= row < 3`: Checks if the row index is within the valid range (0, 1, or 2).
    # 2. `0 <= col < 3`: Checks if the column index is within the valid range (0, 1, or 2).
    # 3. `board[row][col] == 0`: Checks if the selected cell on the board is currently empty.
    # If all three conditions are `True`, the move is valid.
    if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == 0:
        
        # If the move is valid, the board is updated.
        # `board[row][col] = player` places the player's mark (either 1 or -1) at the specified position.
        # This modifies the list of lists 'in-place'.
        board[row][col] = player
        
        # The function returns `True` to indicate that the move was successfully made.
        return True
    
    # If the `if` condition was false, it means the move was invalid (e.g., cell was already taken).
    # The function returns `False` to indicate the move was not made.
    # Without this function, there would be no way to place a mark on the board according to game rules.
    return False

# This function checks if a specific player has won the game.
def check_win(board, player):
    """check if player has won."""
    
    # This comment explains that the function will check rows, columns, and diagonals.
    # check rows, cols, and diags
    
    # This `for` loop iterates through the rows of the board. `range(3)` generates numbers 0, 1, 2.
    # 'r' will be 0, then 1, then 2.
    for r in range(3):
        # The `all()` function returns `True` if all items in an iterable are true.
        # `board[r][c] == player for c in range(3)` is a generator expression. It checks if every cell `c` in the current row `r`
        # is equal to the `player`'s mark. If all 3 cells in a row match, the player has won.
        if all(board[r][c] == player for c in range(3)): return True # Returns True immediately if a winning row is found.

    # This `for` loop iterates through the columns of the board. 'c' will be 0, 1, 2.
    for c in range(3):
        # Similar to the row check, this checks if all cells `r` in the current column `c` belong to the `player`.
        if all(board[r][c] == player for r in range(3)): return True # Returns True if a winning column is found.

    # This checks the main diagonal (top-left to bottom-right).
    # It checks if `board[0][0]`, `board[1][1]`, and `board[2][2]` all belong to the `player`.
    if all(board[i][i] == player for i in range(3)): return True # Returns True for a main diagonal win.

    # This checks the anti-diagonal (top-right to bottom-left).
    # It checks if `board[0][2]`, `board[1][1]`, and `board[2][0]` all belong to the `player`.
    # `2 - i` creates the correct column index for each row `i`.
    if all(board[i][2 - i] == player for i in range(3)): return True # Returns True for an anti-diagonal win.

    # If none of the above conditions were met after checking everything, the player has not won.
    # The function returns `False`.
    # Without this function, the program could not determine when a player has won the game.
    return False

# This function checks if the board is completely filled with moves.
def is_board_full(board):
    """check if board is full (draw)."""
    
    # This line uses a generator expression inside the `all()` function.
    # `for r in range(3) for c in range(3)` is a nested loop that iterates over every cell (r, c) on the board.
    # `board[r][c] != 0` checks if a cell is NOT empty.
    # `all()` checks if this condition is `True` for every single cell. If so, the board is full.
    # This signifies a draw if no one has won.
    # Without this function, the program couldn't identify a draw condition.
    return all(board[r][c] != 0 for r in range(3) for c in range(3))

# This function checks if the game has ended, either by a win or a draw.
def is_match_over(board):
    """check if game has ended."""
    
    # It returns `True` if any of the following conditions are met:
    # 1. `check_win(board, 1)`: Player 'X' (agent) has won.
    # 2. `check_win(board, -1)`: Player 'O' (opponent) has won.
    # 3. `is_board_full(board)`: The board is full, resulting in a draw.
    # The `or` keyword means the expression is true if at least one of the conditions is true.
    # This is a crucial function for controlling the game loop.
    # Without it, the program wouldn't know when to stop a game.
    return check_win(board, 1) or check_win(board, -1) or is_board_full(board)

# This function assigns a numerical reward based on the final state of the board.
# This is a core concept in Reinforcement Learning.
def get_reward(board):
    """get reward based on game outcome."""
    
    # If `check_win(board, 1)` is true, it means the agent (player 1) has won.
    if check_win(board, 1): return 1 # A positive reward of `1` is given for a win.
    
    # If `check_win(board, -1)` is true, it means the opponent (player -1) has won.
    if check_win(board, -1): return -1 # A negative reward (penalty) of `-1` is given for a loss.
    
    # If neither condition is met, it implies a draw.
    return 0 # A neutral reward of `0` is given for a draw.
    # This reward signal is essential for the agent to learn which outcomes are good, bad, or neutral.
    # Without it, the RL algorithm would have no basis for learning.

# This function finds all the possible moves on the current board.
def get_available_moves(board):
    """ret list of empty cells."""
    
    # This uses a list comprehension, which is a concise way to create lists.
    # `for r in range(3) for c in range(3)`: It iterates through every cell of the board.
    # `if board[r][c] == 0`: It includes the cell in the list only if it's empty.
    # `(r, c)`: The coordinates are stored as a tuple.
    # The result is a list of tuples, e.g., `[(0, 0), (1, 2), ...]`, representing all empty spots.
    # This is used by the agent to know what moves it can legally make.
    # Without it, the agent might try to make invalid moves.
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == 0]


# --- REINFORCEMENT LEARNING AGENT CLASS ---
# A class is a blueprint for creating objects. This `RLAgent` class defines the structure
# and behavior of our AI player.

class RLAgent:
    # The `__init__` method is a special method called a constructor. It is executed when a new
    # `RLAgent` object is created. `self` refers to the instance of the object being created.
    def __init__(self, player, epsilon=1.0, min_epsilon=0.1, alpha=0.3, gamma=0.9):
        # These are hyperparameters for the learning algorithm.
        
        # `self.player`: Stores the player's ID (e.g., 1 for 'X'). This lets the agent know which moves are its own.
        self.player = player
        
        # `self.epsilon`: The exploration rate. It's a value between 0 and 1.
        # It determines the probability that the agent will make a random move ("explore") versus
        # making the best move it currently knows ("exploit"). A high epsilon means more exploration.
        # Starts at 1.0 (100% exploration).
        self.epsilon = epsilon
        
        # `self.max_epsilon`: Stores the initial epsilon value to use it later for calculating the decay rate.
        self.max_epsilon = epsilon
        
        # `self.min_epsilon`: The lowest value epsilon is allowed to reach. Prevents the agent from
        # completely stopping exploration, which can be useful.
        self.min_epsilon = min_epsilon
        
        # `self.epsilon_decay`: The rate at which epsilon decreases after each game. This will be calculated later.
        # A decay factor makes the agent explore a lot at the beginning and then exploit its knowledge more as it learns.
        self.epsilon_decay = 1.0
        
        # `self.alpha`: The learning rate. It's a value between 0 and 1.
        # It controls how much the agent adjusts its knowledge based on new experience (a reward).
        # A higher alpha means the agent learns faster but can be more unstable.
        self.alpha = alpha
        
        # `self.gamma`: The discount factor. It's a value between 0 and 1.
        # It determines the importance of future rewards. A value close to 1 makes the agent
        # value long-term rewards more, while a value close to 0 makes it prioritize immediate rewards.
        self.gamma = gamma
        
        # `self.states_value`: This is the agent's "brain" or "policy". It's a dictionary.
        # - The keys of the dictionary are board states. Since dictionary keys must be immutable (unchangeable),
        #   the board (a list of lists) is converted to a tuple of tuples, e.g., `((1, 0, 0), (0, -1, 0), (0, 0, 0))`.
        # - The values are floating-point numbers representing the "value" or "quality" of that state. A higher value
        #   means the agent predicts a better outcome from that state.
        # It is initialized as an empty dictionary `{}`. The agent fills it as it plays and learns.
        # Without this, the agent would have no memory of what it has learned.
        self.states_value = {}

    # This method retrieves the learned value for a given board state from the policy dictionary.
    def get_state_value(self, state):
        """get val of a state from policy."""
        # `self.states_value.get(state, 0.0)` is a safe way to access a dictionary.
        # It tries to find the key `state`. If it finds it, it returns the associated value.
        # If the key is not found (meaning the agent has never seen this state before),
        # it returns the default value `0.0`, implying a neutral or unknown value.
        # This prevents errors and provides a baseline for new states.
        return self.states_value.get(state, 0.0)

    # This method decides which move to make based on the epsilon-greedy policy.
    def choose_action(self, board):
        """choose action using epsilon-greedy policy."""
        # First, get a list of all legal moves.
        available_moves = get_available_moves(board)
        
        # If there are no available moves, the game must be over. Return `None` to indicate no action can be taken.
        if not available_moves:
            return None
        
        # This is the exploration vs. exploitation decision.
        # `random.uniform(0, 1)` generates a random float between 0 and 1.
        if random.uniform(0, 1) < self.epsilon:
            # EXPLORATION: If the random number is less than epsilon, choose a random move.
            # `random.choice(available_moves)` picks one item randomly from the list.
            return random.choice(available_moves) # explore
        
        # EXPLOITATION: If the random number was not less than epsilon, choose the best known move.
        best_move = None # Initialize to None.
        # Initialize `best_value` to negative infinity to ensure that the value of the first evaluated move will be higher.
        best_value = -float('inf')
        # This list will hold all moves that are tied for the best value.
        best_moves = []
        
        # Iterate through every possible move to evaluate them.
        for move in available_moves:
            # To evaluate a move, we need to see what state it leads to.
            # We create a temporary copy of the board to simulate the move.
            # `[row[:] for row in board]` is a deep copy to avoid modifying the actual game board.
            next_board = [row[:] for row in board]
            
            # Make the move on the temporary board.
            make_move(next_board, move[0], move[1], self.player)
            
            # Convert the resulting board state into an immutable tuple to use as a dictionary key.
            next_state = tuple(map(tuple, next_board))
            
            # Look up the value of this potential next state from the agent's memory.
            value = self.get_state_value(next_state)
            
            # Compare this move's value to the best value found so far.
            if value > best_value:
                # If this move is better than any seen before, update the best value.
                best_value = value
                # Store this move as the new best move.
                best_moves = [move]
            elif value == best_value:
                # If this move is just as good as the current best, add it to the list of tied best moves.
                # This prevents the agent from always picking the first best move it finds, adding a bit of randomness.
                best_moves.append(move)
        
        # After checking all moves, decide which one to return.
        if best_moves:
            # If there's at least one best move, choose one randomly from the list of best moves.
            return random.choice(best_moves)
        else:
            # This is a fallback case. If for some reason `best_moves` is empty (e.g., all moves lead to unseen states with value 0),
            # just pick any random available move to prevent an error.
            return random.choice(available_moves)

    # This method updates the agent's policy (the `states_value` dictionary) after a game is complete.
    # It uses the Temporal Difference (TD) learning algorithm.
    def update_state_values(self, states, reward):
        """update state values using TD learning."""
        
        # We iterate through the sequence of states the agent visited during the game in reverse order.
        # This is because the final reward is known, and we propagate this information backward in time.
        for state in reversed(states):
            # Get the current estimated value of the state.
            value = self.get_state_value(state)
            
            # This is the core TD update rule. It adjusts the state's value.
            # `self.gamma * reward`: This is the "target" value. It's the discounted reward from the next state.
            # `self.gamma * reward - value`: This is the "TD error," the difference between the target and the current estimate.
            # `self.alpha * (...)`: We multiply the error by the learning rate to determine how much to adjust the old value.
            # `value += ...`: We update the value by adding this adjustment.
            value += self.alpha * (self.gamma * reward - value)
            
            # We save the newly calculated value back into our policy dictionary.
            self.states_value[state] = value
            
            # Crucially, for the next state in the reversed loop, the "reward" is the updated value of the current state.
            # This is how the value of the final outcome is passed back through all the preceding moves.
            reward = value

    # This method calculates the decay rate for epsilon.
    def calculate_decay(self, total_episodes):
        """calc exponential decay rate for epsilon."""
        # Only calculate if there are episodes to train on.
        if total_episodes > 0:
            # This formula calculates a rate such that epsilon will exponentially decay from its
            # max value to its min value over the course of `total_episodes`.
            # decay_rate = (end_value / start_value)^(1 / number_of_steps)
            self.epsilon_decay = (self.min_epsilon / self.max_epsilon) ** (1 / total_episodes)
        else:
            # If there are no episodes, the decay rate is 1 (meaning no decay).
            self.epsilon_decay = 1.0

    # This method applies the decay to epsilon, usually called after each episode.
    def decay_epsilon(self):
        """apply decay to epsilon."""
        # Multiplies the current epsilon by the decay factor, making it smaller.
        # `max(...)` ensures that epsilon never drops below the specified minimum value.
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# --- TRAINING AND POLICY MANAGEMENT FUNCTIONS ---

# This function controls the training loop for the RL agent.
def train(agent, opponent_player, episodes=4201337): # 4,201,337 is a large, whimsical number of games to train for.
    """train agent via self-play."""
    # First, calculate the epsilon decay rate based on the total number of training games.
    agent.calculate_decay(episodes)

    # This dictionary defines specific episodes at which we want to save a checkpoint of the agent's policy.
    # Saving at different percentages allows us to see how the agent's skill evolves over time.
    save_points = {
        int(episodes * 0.01): "1%",
        int(episodes * 0.05): "5%",
        int(episodes * 0.10): "10%",
        int(episodes * 0.20): "20%",
        int(episodes * 0.50): "50%",
        int(episodes * 0.75): "75%",
        episodes: "100%",
    }

    # The main training loop. It will run for the specified number of 'episodes' (games).
    for episode in range(episodes):
        # At the start of each game, create a new, empty board.
        board = initialize_board()
        # Randomly choose whether the agent (1) or the opponent (-1) goes first. This adds variety to the training.
        current_player = random.choice([1, -1])
        # `agent_states` will store the sequence of board states that result from the agent's moves in this specific game.
        agent_states = []

        # This `while` loop continues as long as the game is not over.
        while not is_match_over(board):
            # If it's the agent's turn...
            if current_player == agent.player:
                # The agent chooses an action (a move).
                action = agent.choose_action(board)
                # If a valid action was returned...
                if action:
                    # Make the move on the board.
                    make_move(board, action[0], action[1], agent.player)
                    # Record the resulting board state by converting it to a tuple and adding it to our list for this episode.
                    agent_states.append(tuple(map(tuple, board)))
            # If it's the opponent's turn...
            else:
                # We double-check the game isn't over before the opponent moves (it could have ended on the agent's last move).
                if not is_match_over(board):
                    # Get all possible moves for the opponent.
                    moves = get_available_moves(board)
                    # If there are moves to be made...
                    if moves:
                        # The opponent makes a random choice. This is a simple but effective way to train.
                        move = random.choice(moves)
                        make_move(board, move[0], move[1], opponent_player)
            
            # Switch turns for the next loop iteration. Multiplying by -1 flips 1 to -1 and -1 to 1.
            current_player *= -1
        
        # After the `while` loop ends, the game is over. Get the final reward.
        reward = get_reward(board)
        # Update the agent's brain using the states from the game and the final reward.
        agent.update_state_values(agent_states, reward)
        
        # After updating, decay epsilon slightly for the next game.
        agent.decay_epsilon()

        # Check if the current episode number (plus 1 for human-readable counting) is one of our save points.
        if (episode + 1) in save_points:
            percentage = save_points[episode + 1] # Get the percentage string (e.g., "50%").
            filename = f"rl_agent_policy_at_{percentage}.pkl" # Create a descriptive filename.
            save_policy(agent, filename) # Call the function to save the policy.
            # Print a message to the console to show progress.
            print(f"Policy saved to {filename} at {percentage} completion ({episode + 1} episodes).")

# This function saves the agent's learned policy to a file using pickle.
def save_policy(agent, filename="rl_agent_policy.pkl"): # A default filename is provided.
    """save agent's states_value dict to a file."""
    # `with open(...)` is the standard way to handle files in Python. It ensures the file is automatically closed.
    # `filename` is the name of the file to be created.
    # `'wb'` means "write binary" mode, which is required for pickle.
    # `as f` gives a name `f` to the file handle.
    with open(filename, 'wb') as f:
        # `pickle.dump()` serializes the object.
        # `agent.states_value` is the object we want to save (the dictionary).
        # `f` is the file to which it will be saved.
        pickle.dump(agent.states_value, f)
    print(f"Policy saved to {filename}") # Confirmation message.
    # Without this function, training progress would be lost every time.

# This function loads a policy from a file into the agent.
def load_policy(agent):
    """load policy from file, trying latest first."""
    # This list defines the order in which to try loading policy files.
    # It starts with the most-trained (100%) and goes down to the least-trained (1%).
    # This ensures we load the best available policy.
    policy_files = [
        "rl_agent_policy_at_100%.pkl", "rl_agent_policy_at_75%.pkl", "rl_agent_policy_at_50%.pkl",
        "rl_agent_policy_at_20%.pkl", "rl_agent_policy_at_10%.pkl", "rl_agent_policy_at_5%.pkl", "rl_agent_policy_at_1%.pkl"
    ]
    # Loop through the list of filenames.
    for filename in policy_files:
        # A `try...except` block is used for error handling.
        try:
            # We attempt to open and read the file in "read binary" (`'rb'`) mode.
            with open(filename, 'rb') as f:
                # `pickle.load(f)` reads the serialized data from the file `f` and reconstructs the Python object.
                # This loaded dictionary is then assigned to the agent's `states_value` attribute.
                agent.states_value = pickle.load(f)
            print(f"Policy loaded from {filename}") # Confirmation message.
            return True # Return `True` to indicate that loading was successful.
        # If the `try` block fails because the file does not exist, a `FileNotFoundError` occurs.
        except FileNotFoundError:
            # The `continue` keyword skips to the next iteration of the loop, trying the next filename.
            continue
    # If the loop finishes without finding any policy file, this message is printed.
    print("No saved policy found. Starting with a new one.")
    # Return `False` to indicate that no policy was loaded.
    return False

# --- GUI CLASS FOR TIC-TAC-TOE ---
# This class manages the entire graphical user interface using tkinter.

class TicTacToeGUI:
    # The constructor for the GUI class.
    # `root` is the main tkinter window. `agent` is the trained RL agent object.
    def __init__(self, root, agent):
        self.root = root # Store the main window instance.
        self.root.title("Tic-Tac-Toe") # Set the title of the window.
        self.agent = agent # Store the agent instance to interact with it.
        self.board = initialize_board() # Create a fresh board for the game.
        self.current_player = 1 # Set the default starting player. This will be changed by user choice.
        # Create a 3x3 list of lists to hold the button widgets, mirroring the board structure.
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.game_over = False # A flag to track if the current game has ended.
        self.start_time = None # Will store the time when the game starts.
        self.status_label = None # Will hold the label widget that displays game status/timer.
        self.choice_frame = None # Will hold the initial frame for choosing who goes first.
        
        # Create the grid of buttons but leave them disabled initially until the player makes a choice.
        self.create_widgets(initial_state='disabled')
        # Display the initial screen to the user.
        self.setup_choice_screen()

    # This method creates the initial screen where the user decides who starts.
    def setup_choice_screen(self):
        """creates screen for player to choose who goes first."""
        # A Frame is a container widget. This one will cover the whole window.
        self.choice_frame = tk.Frame(self.root, bg='grey')
        self.choice_frame.place(relwidth=1, relheight=1) # Makes the frame fill the entire window.

        # A second frame within the first one to center the content.
        content_frame = tk.Frame(self.choice_frame, bg='grey')
        content_frame.pack(expand=True) # `pack` with `expand=True` helps in centering.

        # A text label.
        label = tk.Label(content_frame, text="choose.", font=('normal', 16), bg='grey')
        label.pack(pady=10) # `pady` adds padding on the y-axis.

        # A frame to hold the two buttons side-by-side.
        button_frame = tk.Frame(content_frame, bg='grey')
        button_frame.pack(pady=10)

        # The button to let the agent go first. The text is a joke.
        # `command=lambda: self.start_game(1)`: When clicked, it calls `start_game` with player 1 (agent).
        agent_button = tk.Button(button_frame, text="i want to not win.", command=lambda: self.start_game(1), fg="red")
        agent_button.pack(side=tk.LEFT, padx=5) # `side=tk.LEFT` places it on the left.

        # The button to let the human go first.
        # `command=lambda: self.start_game(-1)`: Calls `start_game` with player -1 (human).
        player_button = tk.Button(button_frame, text="i want to go first.", command=lambda: self.start_game(-1), fg="blue")
        player_button.pack(side=tk.RIGHT, padx=5) # `side=tk.RIGHT` places it on the right.
        
        # A small watermark label.
        watermark = tk.Label(self.choice_frame, text="made by vinz", font=('normal', 8), bg='grey')
        watermark.place(relx=0.5, rely=0.95, anchor=tk.CENTER) # Positions it at the bottom-center.

    # This method is called after the player makes a choice on the start screen.
    def start_game(self, starting_player):
        """starts the game with the selected player."""
        self.current_player = starting_player # Set the starting player.
        self.choice_frame.destroy() # Remove the choice screen widgets.
        self.enable_buttons() # Enable the game board buttons so they can be clicked.
        
        # If the agent is chosen to go first...
        if self.current_player == self.agent.player:
            # `self.root.after(500, ...)` schedules the `agent_move` function to run after a 500ms delay.
            # This makes the agent's move not instantaneous, improving user experience.
            self.root.after(500, self.agent_move)

    # This method creates the main widgets of the game (the grid and the status label).
    def create_widgets(self, initial_state='normal'):
        """create GUI buttons."""
        # Create the status label at the bottom of the grid.
        self.status_label = tk.Label(self.root, text="Time Elapsed: 0s", font=('normal', 12))
        self.status_label.grid(row=3, columnspan=3) # `grid` is a geometry manager for placing widgets.
        
        # Nested loops to create the 3x3 grid of buttons.
        for r in range(3):
            for c in range(3):
                # `lambda r=r, c=c: ...` is important. It captures the current values of `r` and `c`
                # for the `on_button_click` command. Without it, all buttons would use the final values of r and c.
                self.buttons[r][c] = tk.Button(self.root, text='', font=('normal', 40), width=5, height=2,
                                               command=lambda r=r, c=c: self.on_button_click(r, c),
                                               state=initial_state, bg='lightgrey' if initial_state == 'disabled' else 'white')
                # Place the button in the grid at its row and column.
                self.buttons[r][c].grid(row=r, column=c)

    # This method enables all grid buttons.
    def enable_buttons(self):
        """enables all buttons on the grid."""
        for r in range(3):
            for c in range(3):
                # `.config()` is used to change widget options after creation.
                self.buttons[r][c].config(state='normal', bg='white')

    # This method updates the timer on the status label.
    def update_time(self):
        """update elapsed time label."""
        # If the game is over, stop the timer updates.
        if self.game_over:
            return
        # If the game has started...
        if self.start_time:
            # Calculate elapsed seconds.
            elapsed_time = int(time.time() - self.start_time)
            # Update the label's text.
            self.status_label.config(text=f"Time Elapsed: {elapsed_time}s")
        # Schedule this function to run itself again after 1000ms (1 second), creating a loop.
        self.root.after(1000, self.update_time)

    # This method is the event handler for when a human clicks a button.
    def on_button_click(self, r, c):
        """handle human player's move."""
        # If this is the very first move, record the start time and begin the timer.
        if not self.start_time:
            self.start_time = time.time()
            self.update_time()
        
        # Check if the move is valid: the clicked cell must be empty, it must be the human's turn (-1), and the game must not be over.
        if self.board[r][c] == 0 and self.current_player == -1 and not self.game_over:
            # Make the move on the internal board representation.
            make_move(self.board, r, c, -1)
            # Update the GUI to show the new 'O'.
            self.update_board()
            # After the move, check if the game has ended.
            if self.check_game_over():
                return # If the game is over, stop here.
            
            # If the game is not over, switch the current player to the agent (1).
            self.current_player = 1
            # Schedule the agent to make its move after a short delay.
            self.root.after(500, self.agent_move)

    # This method handles the logic for the agent's turn.
    def agent_move(self):
        """handle agent's move."""
        # Start the timer if it hasn't been started (in case the agent moves first).
        if not self.start_time:
            self.start_time = time.time()
            self.update_time()
            
        # Check that it's the agent's turn and the game isn't over.
        if self.current_player == self.agent.player and not self.game_over:
            # Ask the agent to choose an action based on the current board.
            action = self.agent.choose_action(self.board)
            # If the agent returned a valid move...
            if action:
                # Make the move on the internal board.
                make_move(self.board, action[0], action[1], self.agent.player)
                # Update the GUI to show the new 'X'.
                self.update_board()
                # Check if the game is over after the agent's move.
                if self.check_game_over():
                    return # Stop if the game is over.
                # Switch the current player back to the human (-1).
                self.current_player = -1

    # This method synchronizes the GUI display with the internal board state.
    def update_board(self):
        """update GUI board display."""
        for r in range(3):
            for c in range(3):
                # If the board cell has a 1, display 'X' in red.
                if self.board[r][c] == 1:
                    # Set the button's text, disable it, and set the color of the disabled text.
                    self.buttons[r][c].config(text='X', state='disabled', disabledforeground='red')
                # If the board cell has a -1, display 'O' in blue.
                elif self.board[r][c] == -1:
                    self.buttons[r][c].config(text='O', state='disabled', disabledforeground='blue')

    # This method checks for a win or draw and updates the game state.
    def check_game_over(self):
        """check for game over and display result."""
        # Check if Player 'X' (agent) won.
        if check_win(self.board, 1):
            self.status_label.config(text="X wins! Thanks for playing! :)")
            self.game_over = True # Set the game over flag.
            return True # Return True indicating the game has ended.
        # Check if Player 'O' (human) won.
        elif check_win(self.board, -1):
            self.status_label.config(text="O wins! Thanks for playing! :)")
            self.game_over = True
            return True
        # Check for a draw.
        elif is_board_full(self.board):
            self.status_label.config(text="It's a draw! Thanks for playing! :)")
            self.game_over = True
            return True
        # If none of the above, the game is still ongoing.
        return False

# --- SCRIPT ENTRY POINT ---
# The `if __name__ == "__main__":` block is a standard Python construct.
# The code inside this block will only run when the Python script is executed directly,
# not when it is imported as a module into another script. This is the starting point of the program.

if __name__ == "__main__":
    # Create an instance of our RL agent. The agent will play as player `1` ('X').
    rl_agent = RLAgent(player=1)
    
    # Try to load a pre-trained policy from a file. `load_policy` returns True on success, False on failure.
    # The `if not ...` means "if loading failed...".
    if not load_policy(rl_agent):
        # If no policy was found, we need to train a new one.
        print("Training the RL agent...")
        # Call the `train` function to start the training process. The agent (player 1) will play
        # against a random opponent (player -1) for the specified number of episodes.
        train(rl_agent, opponent_player=-1)
        print("Training finished.")

    # After loading or training, we are ready to play against the human.
    # For gameplay, we want the agent to always use its best knowledge, so we turn off exploration.
    # Setting epsilon to 0 means the agent will always exploit (never make a random move).
    rl_agent.epsilon = 0 
    
    # Create the main window for the tkinter application.
    root = tk.Tk()
    # Create an instance of our GUI class, passing it the main window and the trained agent.
    gui = TicTacToeGUI(root, rl_agent)
    # `root.mainloop()` starts the tkinter event loop. This line is crucial.
    # It listens for events like button clicks and keeps the window open until the user closes it.
    # The program will effectively pause here, running the GUI until it's closed.
    root.mainloop()
