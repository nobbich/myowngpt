#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:32:15 2025

@author: nsuter
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# Modell für Embeddings laden
model = SentenceTransformer("all-MiniLM-L6-v2")

# Funktion zum Laden der Datei und Umwandeln in Embeddings
def load_file_and_add_to_faiss(file_path, index):
    try:
        # Datei lesen (angenommen, es ist eine Textdatei)
        with open(file_path, 'r', encoding='utf-8') as file:
            documents = file.readlines()
        
        # Texte in Vektoren umwandeln
        vectors = np.array(model.encode(documents), dtype=np.float32)
        
        # Vektoren zum FAISS-Index hinzufügen
        index.add(vectors)
        return documents
    except Exception as e:
        messagebox.showerror("Fehler", f"Fehler beim Laden der Datei: {e}")
        return None

# Funktion zum Öffnen des Datei-Dialogs und Auswählen der Datei
def open_file():
    file_path = filedialog.askopenfilename(title="Wählen Sie eine Datei zum Importieren")
    if file_path:
        selected_index = index_listbox.curselection()
        if not selected_index:
            messagebox.showwarning("Keine Auswahl", "Bitte wählen Sie einen FAISS-Index.")
            return
        
        selected_index = selected_index[0]
        faiss_index = faiss_indexes[selected_index]

        # Datei laden und in den ausgewählten FAISS-Index hinzufügen
        documents = load_file_and_add_to_faiss(file_path, faiss_index)
        if documents:
            messagebox.showinfo("Erfolg", f"{len(documents)} Dokumente hinzugefügt.")
            refresh_index_listbox()

# Funktion zum Erstellen eines neuen FAISS-Index
def create_new_index():
    dimension = 384  # Beispiel-Dimension (z.B. für MiniLM)
    new_index = faiss.IndexFlatL2(dimension)
    
    # Überprüfen, ob ein GPU verfügbar ist
    if faiss.get_num_gpus() > 0:
        # GPU ist verfügbar, Index auf GPU verschieben
        new_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, new_index)
        messagebox.showinfo("Erfolg", "FAISS-Index wurde auf die GPU verschoben.")
    else:
        # Kein GPU, nur CPU-basierter Index
        messagebox.showinfo("Info", "Kein GPU gefunden, der Index wird auf der CPU verwendet.")
    
    faiss_indexes.append(new_index)
    refresh_index_listbox()

# Funktion zum Aktualisieren der Anzeige der FAISS-Indizes
def refresh_index_listbox():
    index_listbox.delete(0, tk.END)
    for i, faiss_index in enumerate(faiss_indexes):
        index_listbox.insert(tk.END, f"FAISS Index {i+1}")

# Fenster für die GUI
root = tk.Tk()
root.title("FAISS Index Management")

# Listbox für die Auswahl des FAISS-Indexes
index_listbox = tk.Listbox(root, height=10, width=50)
index_listbox.pack(pady=20)

# Button zum Erstellen eines neuen Index
create_button = tk.Button(root, text="Neuen FAISS-Index erstellen", command=create_new_index)
create_button.pack(pady=5)

# Button zum Öffnen einer Datei und Hinzufügen zu einem FAISS-Index
open_button = tk.Button(root, text="Datei zum Index hinzufügen", command=open_file)
open_button.pack(pady=5)

# Initialisieren des FAISS-Indexes und der Indizes-Liste
faiss_indexes = []

# Hauptloop der GUI
root.mainloop()
