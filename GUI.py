# GUI Tkinter window
# Create an interactive GUI that allows users to enter the desired qualities of a post-secondary institution and generate multiple recommendations

import tkinter as tk
import tkinter.ttk as ttk
from models import Models

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Post-Secondary Institution Recommendation System")
        self.root.geometry("800x600")

    def label(self, text, col, row):
        self.label = tk.Label(self.root, text=text)
        self.label.grid(column=col, row=row, padx=10, pady=10)

    def entry(self, col, row):
        self.entry = tk.Entry(self.root)
        self.entry.grid(column=col, row=row, padx=10, pady=10)

    def button(self, text, col, row, command):
        self.button = tk.Button(self.root, text=text, command=command)
        self.button.grid(column=col, row=row, padx=10, pady=10)

    def generate_model(self):
        model = Models()
        model.gen()

GUI = GUI() # Not fully ready yet