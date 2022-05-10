import random
from math import log,sqrt,e,inf
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
from cairosvg import svg2png
import chess
import chess.pgn
import chess.svg
import chess.engine
import streamlit as st


class Node():
    def __init__(self):
        self.state = chess.Board()
        self.children = set()
        self.root = None
        self.parent_visits = 0
        self.child_visits = 0
        self.score = 0

def ucb(current):
    ucb=current.score + 2*(sqrt(log(current.parent_visits+e+(10**-8))/(current.child_visits+(10**-10))))
    return ucb

# Selects node with the highest ucb value
def select(current, white):
    best_child = None
    if white:
        max_ucb = -inf
        for i in current.children:
            ucb_i = ucb(i)
            if ucb_i>max_ucb:
                max_ucb = ucb_i
                best_child = i
    else:
        min_ucb = inf
        for i in current.children:
            ucb_i = ucb(i)
            if ucb_i<min_ucb:
                min_ucb = ucb_i
                best_child = i
    return best_child

# Keep selecting best child/state from appropriate color's states and move down the tree until a leaf node is reached.
def expand(current,white):
    if len(current.children)==0:
        return current
    best_child=select(current,white)
    if white:
        return expand(best_child,0)
    else:
        return expand(best_child,1)

# Keep playing until game is over
def simulate(current):
    # Given leaf node from expand step, if game is over give reward.
    if current.state.is_game_over():
        if current.state.result()=='1-0':
            return (1,current)
        elif current.state.result()=='0-1': 
            return (-1,current)
        else:
            return (0.5,current)
    
    # Get all possible moves from this new board state
    possible_moves = [current.state.san(i) for i in list(current.state.legal_moves)]
    # Get children boards of this board, and pick a random one to continue to simulate
    for move in possible_moves:
        temp_state = chess.Board(current.state.fen())
        temp_state.push_san(move)
        child = Node()
        child.state = temp_state
        child.root = current
        current.children.add(child)
    rand_state = random.choice(list(current.children))

    return simulate(rand_state)


# Update number of times child and parent states have been visited(later used in ucb).
# Add reward to score for this end game board.Go back up the path taken to reach the leaf node, 
# update number of visits and reward of each node along the path.

def backpropagation(current,reward):
    current.child_visits+=1
    while current.root!=None:
        current.score+=reward
        current.parent_visits+=1
        current = current.root
    current.score+=reward
    return current

# Gives the possible results of that board as children nodes of the current node
def init_round(possible_moves,current):
    states_moves=dict()
    for move in possible_moves:
        # Gives state of board after the move of i(ex : e6)
        temp_state = chess.Board(current.state.fen())
        temp_state.push_san(move)
        # Create node for each board generated from the moves. Add this board as a child of the current board
        res = Node()
        res.state = temp_state
        res.root = current
        current.children.add(res)
        # Store node generated as key and the move that led to it as the value
        states_moves[res] = move
    return states_moves

# At end of mcts round, calculate ucb with updated visit values and select move with max ucb.
def select_move(current,states_moves,white):
    selected_move = ''
    if white:
        max_ucb = -inf
        for i in current.children:
            ucb_i = ucb(i)
            if ucb_i>max_ucb:
                max_ucb = ucb_i
                selected_move = states_moves[i]
    else:
        min_ucb = inf
        for i in current.children:
            ucb_i = ucb(i)
            if ucb_i<min_ucb:
                min_ucb = ucb_i
                selected_move = states_moves[i]
    return selected_move


def mcts(current,white,i,over):
    if over:
        return -1
    # Get all possible moves in this board state for the pieces that can move, and generate tree of possible moves
    possible_moves = [current.state.san(i) for i in list(current.state.legal_moves)]
    states_moves = init_round(possible_moves,current)

    while i>0:
        # Gets child node/board that has the highest ucb value
        best_child=select(current,white)

        if white:
            ex_child = expand(best_child,0)
        else:
            ex_child = expand(best_child,1)

        reward,state = simulate(ex_child)
        current = backpropagation(state,reward)
        i-=1
    
    selected_move=select_move(current,states_moves,white)
    return selected_move
    


st.header("A Game of Chess using Monte Carlo Tree Search")
st.sidebar.header("Group 7 - Chess using Monte Carlo Tree Search")
st.sidebar.write("To make each move, it uses the MCTS algorithm, which consists of 4 parts: Selection, Expansion, Simulation, and Backpropagation. \n Selection: Selects the best child node based on the UCB value. \n Expansion: Generates the possible moves for the pieces that can move, and adds them as children nodes of the current node. \n Simulation: Simulates the game until the game is over.      Backpropagation: Updates the visit values of the nodes along the path taken to reach the leaf node, and adds the reward to the score of the node. \n \n")
board = chess.Board()
png = svg2png(bytestring=chess.svg.board(board,colors={"square light":"#ffcc9c", "square dark":"#d88c44"}))
p1 = st.empty()
imageLocation = st.empty()
p3 = st.empty()
p2 = st.empty()

# Open png in PIL
pil_img = Image.open(BytesIO(png)).convert('RGBA')
imageLocation.image(pil_img, caption='Initial pose', width=600)
white = 1 # white - 1, black - 0
moves = 0
pgn = []
game = chess.pgn.Game()

df = pd.DataFrame(columns=['Turn','Moves by White','Moves by Black'])
iteration=1

if st.button("Start"):
    if st.button("stop game"):
        st.stop()
    while not board.is_game_over():
        print("Iteration",iteration)
        iteration+=1
        l  = 'Total number of moves: '+str(moves)
        p3.subheader(l)
        # Store current state of board
        root = Node()
        root.state = board
        if white:
            p1.subheader("White's Turn")
            move = mcts(root,white,10,board.is_game_over())
            df.loc[len(df.index)] = [iteration-1, move, 'No move'] 
        else:
            p1.subheader("Black's Turn")
            move = mcts(root,white,10,board.is_game_over())
            df.loc[len(df.index)] = [iteration-1, 'No move',move] 
        board.push_san(move)

        
        p2.table(df)
        png = svg2png(bytestring=chess.svg.board(board,colors={"square light":"#ffcc9c", "square dark":"#d88c44"}))
        
        pil_img = Image.open(BytesIO(png)).convert('RGBA')
        imageLocation.image(pil_img, caption="Move "+move,width=600)

        pgn.append(move)
        white ^= 1
        moves+=1
    
    if board.result()=="1-0":
        p2.subheader("WHITE WON THE GAME!")
    elif board.result()=="0-1":
        p2.subheader("BLACK WON THE GAME!")
    else:
        p2.subheader("DRAW!")