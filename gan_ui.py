# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:38:34 2024

@author: UKKASHA
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

class GANApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GAN Generator")
        self.root.geometry("400x400")  # Enlarging the window

        self.load_model_button = ttk.Button(self.root, text="Load Generator Model", command=self.load_model)
        self.load_model_button.pack(pady=10)

        self.generate_button = ttk.Button(self.root, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=10)

        self.save_button = ttk.Button(self.root, text="Save Image", command=self.save_image)
        self.save_button.pack(pady=10)

        self.image_label = ttk.Label(self.root)
        self.image_label.pack()

        self.model = None

    def load_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.h5")])
        if file_path:
            self.model = load_model(file_path)
            tk.messagebox.showinfo("Info", "Model loaded successfully!")

    def generate_image(self):
        if self.model:
            try:
                noise = np.random.normal(0, 1, (1, 100))
                generated_image = self.model.predict(noise)
                generated_image = (generated_image.squeeze() + 1) * 127.5
                generated_image = Image.fromarray(generated_image.astype(np.uint8))
                generated_image = generated_image.resize((200, 200))  # Enlarging the image
                photo = ImageTk.PhotoImage(generated_image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo
            except ValueError:
                tk.messagebox.showerror("Error", "Error generating image!")
        else:
            tk.messagebox.showerror("Error", "Please load a model first!")

    def save_image(self):
        if hasattr(self, "image_label") and self.image_label["image"]:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                image = Image.open(self.image_label.image)
                image.save(file_path)
                tk.messagebox.showinfo("Info", "Image saved successfully!")
        else:
            tk.messagebox.showerror("Error", "No image to save!")

def main():
    root = tk.Tk()
    app = GANApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
