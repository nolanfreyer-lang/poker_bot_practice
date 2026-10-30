"""Play PLO4 against the 7 bots, from any folder: python3 play_plo4.py

Takes the same options as PLO4_Poker_Bot/play_vs_bot.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "PLO4_Poker_Bot"))

from play_vs_bot import main

if __name__ == "__main__":
    main()
