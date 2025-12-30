"""
Application Interactive MNIST avec Pygame
Groupe 11 - Deep Learning Demo
"""

import pygame
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import threading

# Désactiver les warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialisation Pygame
pygame.init()

# Constantes
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
CANVAS_SIZE = 400
FPS = 60

# Couleurs stylées
BG_COLOR = (15, 23, 42)          # Bleu foncé
CANVAS_BG = (30, 41, 59)         # Gris bleu
DRAW_COLOR = (255, 255, 255)     # Blanc
BUTTON_COLOR = (59, 130, 246)    # Bleu vif
BUTTON_HOVER = (96, 165, 250)    # Bleu plus clair
TEXT_COLOR = (248, 250, 252)     # Blanc cassé
SUCCESS_COLOR = (34, 197, 94)    # Vert
WARNING_COLOR = (251, 146, 60)   # Orange
PROGRESS_BG = (51, 65, 85)       # Gris foncé
PROGRESS_FILL = (34, 197, 94)    # Vert

# Police
pygame.font.init()
FONT_LARGE = pygame.font.Font(None, 48)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_SMALL = pygame.font.Font(None, 28)
FONT_TINY = pygame.font.Font(None, 20)

class Button:
    def __init__(self, x, y, width, height, text, color=BUTTON_COLOR):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover = False
        
    def draw(self, screen):
        color = BUTTON_HOVER if self.hover else self.color
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        text_surf = FONT_SMALL.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
        
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)

class MNISTApp:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("MNIST Demo - Groupe 11")
        self.clock = pygame.time.Clock()
        
        # États
        self.state = "welcome"  # welcome, training, ready, predicting
        self.running = True
        
        # Canvas de dessin
        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.drawing = False
        self.brush_size = 20
        
        # Modèle
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
        # Entraînement
        self.training_progress = 0
        self.training_epoch = 0
        self.training_accuracy = 0
        self.training_thread = None
        
        # Prédiction
        self.prediction = None
        self.confidence = 0
        
        # Boutons
        self.setup_buttons()
        
    def setup_buttons(self):
        center_x = WINDOW_WIDTH // 2
        self.train_button = Button(center_x - 150, 400, 300, 60, 
                                   "Entraîner le Modèle", SUCCESS_COLOR)
        self.predict_button = Button(50, 550, 200, 50, "Prédire")
        self.clear_button = Button(270, 550, 200, 50, "Effacer", WARNING_COLOR)
        
    def load_data(self):
        """Charge le dataset MNIST"""
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        
    def create_model(self):
        """Crée le réseau de neurones"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model_thread(self):
        """Entraîne le modèle dans un thread séparé"""
        self.state = "training"
        self.load_data()
        self.model = self.create_model()
        
        x_train_flat = self.x_train.reshape(-1, 784)
        
        # Callback personnalisé pour mettre à jour la progression
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, app):
                self.app = app
                
            def on_epoch_end(self, epoch, logs=None):
                self.app.training_epoch = epoch + 1
                self.app.training_progress = ((epoch + 1) / 5) * 100
                self.app.training_accuracy = logs['accuracy'] * 100
        
        self.model.fit(
            x_train_flat,
            self.y_train,
            epochs=5,
            batch_size=128,
            validation_split=0.2,
            verbose=0,
            callbacks=[ProgressCallback(self)]
        )
        
        self.state = "ready"
    
    def start_training(self):
        """Lance l'entraînement dans un thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self.train_model_thread)
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def draw_on_canvas(self, pos):
        """Dessine sur le canvas"""
        canvas_x = 50
        canvas_y = 100
        
        if canvas_x <= pos[0] <= canvas_x + CANVAS_SIZE and \
           canvas_y <= pos[1] <= canvas_y + CANVAS_SIZE:
            x = pos[0] - canvas_x
            y = pos[1] - canvas_y
            
            # Dessiner un cercle épais
            for i in range(-self.brush_size, self.brush_size):
                for j in range(-self.brush_size, self.brush_size):
                    if i*i + j*j <= self.brush_size * self.brush_size:
                        px, py = x + i, y + j
                        if 0 <= px < CANVAS_SIZE and 0 <= py < CANVAS_SIZE:
                            self.canvas[py, px] = 255
    
    def clear_canvas(self):
        """Efface le canvas"""
        self.canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        self.prediction = None
        self.confidence = 0
    
    def predict_digit(self):
        """Fait une prédiction sur le dessin"""
        if self.model is None:
            return
        
        # Redimensionner à 28x28
        img = Image.fromarray(self.canvas)
        img = img.resize((28, 28), Image.LANCZOS)
        img_array = np.array(img) / 255.0
        img_flat = img_array.reshape(1, 784)
        
        # Prédiction
        prediction = self.model.predict(img_flat, verbose=0)
        self.prediction = np.argmax(prediction)
        self.confidence = prediction[0][self.prediction] * 100
    
    def draw_welcome(self):
        """Écran d'accueil"""
        self.screen.fill(BG_COLOR)
        
        # Titre
        title = FONT_LARGE.render("MNIST - Deep Learning Demo", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        
        # Sous-titre
        subtitle = FONT_MEDIUM.render("Reconnaissance de Chiffres Manuscrits", True, (148, 163, 184))
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH // 2, 220))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Info groupe
        group = FONT_SMALL.render("Groupe 11 : DEGBEY, DOSSOU, DOUFFAN, LOGOSSOU", True, (148, 163, 184))
        group_rect = group.get_rect(center=(WINDOW_WIDTH // 2, 300))
        self.screen.blit(group, group_rect)
        
        # Bouton d'entraînement
        self.train_button.draw(self.screen)
        
        # Instructions
        instructions = [
            "1. Cliquez pour entraîner le modèle (~2 minutes)",
            "2. Dessinez un chiffre avec la souris",
            "3. Cliquez 'Prédire' pour voir le résultat"
        ]
        y = 500
        for inst in instructions:
            text = FONT_TINY.render(inst, True, (148, 163, 184))
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, y))
            self.screen.blit(text, text_rect)
            y += 30
    
    def draw_training(self):
        """Écran d'entraînement"""
        self.screen.fill(BG_COLOR)
        
        # Titre
        title = FONT_LARGE.render("Entraînement en cours...", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        
        # Époque
        epoch_text = FONT_MEDIUM.render(f"Époque {self.training_epoch}/5", True, (148, 163, 184))
        epoch_rect = epoch_text.get_rect(center=(WINDOW_WIDTH // 2, 250))
        self.screen.blit(epoch_text, epoch_rect)
        
        # Barre de progression
        bar_width = 600
        bar_height = 40
        bar_x = (WINDOW_WIDTH - bar_width) // 2
        bar_y = 320
        
        pygame.draw.rect(self.screen, PROGRESS_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=20)
        
        if self.training_progress > 0:
            fill_width = int((self.training_progress / 100) * bar_width)
            pygame.draw.rect(self.screen, PROGRESS_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=20)
        
        # Pourcentage
        percent = FONT_MEDIUM.render(f"{int(self.training_progress)}%", True, TEXT_COLOR)
        percent_rect = percent.get_rect(center=(WINDOW_WIDTH // 2, bar_y + bar_height // 2))
        self.screen.blit(percent, percent_rect)
        
        # Précision
        if self.training_accuracy > 0:
            acc_text = FONT_SMALL.render(f"Précision : {self.training_accuracy:.1f}%", True, SUCCESS_COLOR)
            acc_rect = acc_text.get_rect(center=(WINDOW_WIDTH // 2, 420))
            self.screen.blit(acc_text, acc_rect)
        
        # Info
        info = FONT_TINY.render("Le modèle apprend à reconnaître les chiffres...", True, (148, 163, 184))
        info_rect = info.get_rect(center=(WINDOW_WIDTH // 2, 500))
        self.screen.blit(info, info_rect)
    
    def draw_ready(self):
        """Écran de dessin et prédiction"""
        self.screen.fill(BG_COLOR)
        
        # Titre
        title = FONT_MEDIUM.render("Dessinez un chiffre", True, TEXT_COLOR)
        self.screen.blit(title, (50, 30))
        
        # Canvas
        canvas_surface = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        canvas_surface.fill(CANVAS_BG)
        
        # Convertir le canvas numpy en surface Pygame
        for y in range(CANVAS_SIZE):
            for x in range(CANVAS_SIZE):
                color = (self.canvas[y, x], self.canvas[y, x], self.canvas[y, x])
                canvas_surface.set_at((x, y), color)
        
        self.screen.blit(canvas_surface, (50, 100))
        
        # Bordure du canvas
        pygame.draw.rect(self.screen, (71, 85, 105), (50, 100, CANVAS_SIZE, CANVAS_SIZE), 3, border_radius=5)
        
        # Boutons
        self.predict_button.draw(self.screen)
        self.clear_button.draw(self.screen)
        
        # Zone de prédiction
        pred_x = 550
        pred_y = 100
        
        # Titre prédiction
        pred_title = FONT_MEDIUM.render("Prédiction", True, TEXT_COLOR)
        self.screen.blit(pred_title, (pred_x, pred_y))
        
        if self.prediction is not None:
            # Carré de prédiction
            pred_box_y = pred_y + 80
            pygame.draw.rect(self.screen, CANVAS_BG, (pred_x, pred_box_y, 350, 200), border_radius=10)
            
            # Chiffre prédit
            digit = FONT_LARGE.render(str(self.prediction), True, SUCCESS_COLOR)
            digit_rect = digit.get_rect(center=(pred_x + 175, pred_box_y + 70))
            self.screen.blit(digit, digit_rect)
            
            # Confiance
            conf_text = FONT_SMALL.render(f"Confiance : {self.confidence:.1f}%", True, TEXT_COLOR)
            conf_rect = conf_text.get_rect(center=(pred_x + 175, pred_box_y + 150))
            self.screen.blit(conf_text, conf_rect)
        else:
            # Message
            msg = FONT_SMALL.render("Dessinez puis cliquez 'Prédire'", True, (148, 163, 184))
            self.screen.blit(msg, (pred_x, pred_y + 100))
        
        # Instructions
        inst_y = 630
        inst = FONT_TINY.render("Maintenez le bouton gauche de la souris pour dessiner", True, (148, 163, 184))
        inst_rect = inst.get_rect(center=(WINDOW_WIDTH // 2, inst_y))
        self.screen.blit(inst, inst_rect)
    
    def handle_events(self):
        """Gère les événements"""
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Clic gauche
                    if self.state == "welcome":
                        if self.train_button.is_clicked(mouse_pos):
                            self.start_training()
                    
                    elif self.state == "ready":
                        if self.predict_button.is_clicked(mouse_pos):
                            self.predict_digit()
                        elif self.clear_button.is_clicked(mouse_pos):
                            self.clear_canvas()
                        else:
                            self.drawing = True
                            self.draw_on_canvas(mouse_pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.state == "ready":
                    if self.drawing:
                        self.draw_on_canvas(mouse_pos)
                    
                    # Hover des boutons
                    self.predict_button.update_hover(mouse_pos)
                    self.clear_button.update_hover(mouse_pos)
                
                elif self.state == "welcome":
                    self.train_button.update_hover(mouse_pos)
    
    def run(self):
        """Boucle principale"""
        while self.running:
            self.handle_events()
            
            # Dessiner selon l'état
            if self.state == "welcome":
                self.draw_welcome()
            elif self.state == "training":
                self.draw_training()
            elif self.state == "ready":
                self.draw_ready()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    app = MNISTApp()
    app.run()