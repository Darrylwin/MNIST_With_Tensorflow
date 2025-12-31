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
import json
import base64

# Désactiver les warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fichier de sauvegarde des exemples
LEARNING_DATA_FILE = "learned_examples.npz"

# Initialisation Pygame
pygame.init()

# Constantes
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700
CANVAS_SIZE = 400
FPS = 60

# Couleurs stylées
BG_COLOR = (15, 23, 42)          # Bleu foncé
CANVAS_BG = (255, 255, 255)      # Blanc (comme MNIST)
DRAW_COLOR = (0, 0, 0)           # Noir pour dessiner
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
        self.show_digit_selection = False  # Pour afficher les boutons 0-9
        
        # Canvas de dessin (blanc = 255 pour correspondre à MNIST)
        self.canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255
        self.drawing = False
        self.brush_size = 8  # Pinceau plus fin
        
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
        self.training_loss = 0
        self.val_accuracy = 0
        self.training_thread = None
        self.history_accuracy = []
        self.history_val_accuracy = []
        self.history_loss = []
        self.sample_images = None  # Images d'exemple pour l'entraînement
        
        # Prédiction
        self.prediction = None
        self.confidence = 0
        self.waiting_feedback = False  # Attend le feedback de l'utilisateur
        self.learning_examples = []  # Exemples pour l'apprentissage interactif
        self.learning_count = 0  # Nombre d'exemples appris
        self.current_drawing = None  # Dessin actuel pour le feedback
        self.saved_examples_count = 0  # Exemples chargés depuis le fichier
        
        # Charger les exemples sauvegardés
        self.load_learned_examples()
        
        # Boutons
        self.setup_buttons()
        
    def setup_buttons(self):
        center_x = WINDOW_WIDTH // 2
        self.train_button = Button(center_x - 150, 400, 300, 60, 
                                   "Entraîner le Modèle", SUCCESS_COLOR)
        self.predict_button = Button(50, 550, 200, 50, "Prédire")
        self.clear_button = Button(270, 550, 200, 50, "Effacer", WARNING_COLOR)
        
        # Boutons de feedback
        self.correct_button = Button(550, 450, 160, 50, "✅ Correct", SUCCESS_COLOR)
        self.wrong_button = Button(730, 450, 160, 50, "❌ Mauvais", WARNING_COLOR)
        
        # Boutons pour choisir le vrai chiffre (0-9)
        self.digit_buttons = []
        for i in range(10):
            x = 550 + (i % 5) * 70
            y = 520 if i < 5 else 580
            self.digit_buttons.append(Button(x, y, 60, 50, str(i)))
        
    def load_data(self):
        """Charge le dataset MNIST"""
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        
        # Sélectionner UN exemple de CHAQUE chiffre (0-9) pour l'affichage
        self.sample_images = []
        self.sample_labels = []
        
        for digit in range(10):
            # Trouver le premier exemple de ce chiffre
            idx = np.where(self.y_train == digit)[0][0]
            self.sample_images.append((self.x_train[idx] * 255).astype(np.uint8))
            self.sample_labels.append(digit)
        
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
        y_train = self.y_train.copy()
        
        # Ajouter les exemples sauvegardés au dataset d'entraînement
        if len(self.learning_examples) > 0:
            learned_X = np.vstack([ex[0] for ex in self.learning_examples])
            learned_y = np.array([ex[1] for ex in self.learning_examples])
            
            x_train_flat = np.vstack([x_train_flat, learned_X])
            y_train = np.concatenate([y_train, learned_y])
            
            print(f"🔄 Ajout de {len(self.learning_examples)} exemples sauvegardés à l'entraînement")
        
        # Callback personnalisé pour mettre à jour la progression
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, app):
                self.app = app
                
            def on_epoch_end(self, epoch, logs=None):
                self.app.training_epoch = epoch + 1
                self.app.training_progress = ((epoch + 1) / 5) * 100
                self.app.training_accuracy = logs['accuracy'] * 100
                self.app.training_loss = logs['loss']
                self.app.val_accuracy = logs.get('val_accuracy', 0) * 100
                
                # Stocker l'historique pour les graphiques
                self.app.history_accuracy.append(logs['accuracy'] * 100)
                self.app.history_val_accuracy.append(logs.get('val_accuracy', 0) * 100)
                self.app.history_loss.append(logs['loss'])
        
        self.model.fit(
            x_train_flat,
            y_train,
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
            
            # Dessiner un cercle noir (0 = noir comme MNIST)
            for i in range(-self.brush_size, self.brush_size):
                for j in range(-self.brush_size, self.brush_size):
                    if i*i + j*j <= self.brush_size * self.brush_size:
                        px, py = x + i, y + j
                        if 0 <= px < CANVAS_SIZE and 0 <= py < CANVAS_SIZE:
                            self.canvas[py, px] = 0  # Noir
    
    def clear_canvas(self):
        """Efface le canvas"""
        self.canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8) * 255  # Blanc
        self.prediction = None
        self.confidence = 0
        self.waiting_feedback = False
        self.current_drawing = None
    
    def mark_correct(self):
        """Marque la prédiction comme correcte"""
        if self.current_drawing is not None and self.prediction is not None:
            # Ajouter l'exemple aux données d'apprentissage
            self.learning_examples.append((self.current_drawing, self.prediction))
            self.learning_count += 1
            self.waiting_feedback = False
            
            # Sauvegarder automatiquement
            self.save_learned_examples()
            
            print(f"✅ Exemple confirmé : {self.prediction}")
    
    def mark_wrong(self, correct_digit):
        """Marque la prédiction comme fausse et apprend le bon chiffre"""
        if self.current_drawing is not None:
            # Ajouter l'exemple avec le bon label
            self.learning_examples.append((self.current_drawing, correct_digit))
            self.learning_count += 1
            
            # Sauvegarder automatiquement
            self.save_learned_examples()
            
            # Réentraîner rapidement sur les nouveaux exemples
            if len(self.learning_examples) >= 1:
                self.retrain_on_feedback()
            
            self.waiting_feedback = False
            print(f"❌ Correction : {self.prediction} → {correct_digit}")
    
    def retrain_on_feedback(self):
        """Réentraîne le modèle sur les exemples de feedback"""
        if len(self.learning_examples) == 0:
            return
        
        # Préparer les données
        X = np.vstack([ex[0] for ex in self.learning_examples])
        y = np.array([ex[1] for ex in self.learning_examples])
        
        # Réentraîner rapidement (1 époque avec les nouveaux exemples)
        self.model.fit(X, y, epochs=1, batch_size=len(X), verbose=0)
        
        print(f"🔄 Modèle réentraîné sur {len(self.learning_examples)} exemples")
    
    def save_learned_examples(self):
        """Sauvegarde les exemples appris dans un fichier"""
        if len(self.learning_examples) == 0:
            return
        
        try:
            X = np.vstack([ex[0] for ex in self.learning_examples])
            y = np.array([ex[1] for ex in self.learning_examples])
            
            np.savez(LEARNING_DATA_FILE, X=X, y=y)
            print(f"💾 {len(self.learning_examples)} exemples sauvegardés")
        except Exception as e:
            print(f"❌ Erreur de sauvegarde : {e}")
    
    def load_learned_examples(self):
        """Charge les exemples sauvegardés depuis le fichier"""
        if not os.path.exists(LEARNING_DATA_FILE):
            print("📂 Aucun exemple sauvegardé trouvé")
            return
        
        try:
            data = np.load(LEARNING_DATA_FILE)
            X = data['X']
            y = data['y']
            
            # Reconstruire la liste d'exemples
            for i in range(len(X)):
                self.learning_examples.append((X[i:i+1], y[i]))
            
            self.saved_examples_count = len(self.learning_examples)
            self.learning_count = self.saved_examples_count
            print(f"📚 {self.saved_examples_count} exemples chargés depuis les sessions précédentes")
        except Exception as e:
            print(f"❌ Erreur de chargement : {e}")
    
    def predict_digit(self):
        """Fait une prédiction sur le dessin"""
        if self.model is None:
            return
        
        # Inverser les couleurs (MNIST = blanc sur noir, nous avons noir sur blanc)
        inverted = 255 - self.canvas
        
        # Trouver la bounding box du dessin pour le centrer
        coords = np.column_stack(np.where(inverted > 0))
        if len(coords) == 0:
            return  # Rien n'est dessiné
        
        # Obtenir les limites
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Ajouter une marge
        margin = 20
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(CANVAS_SIZE, y_max + margin)
        x_max = min(CANVAS_SIZE, x_max + margin)
        
        # Extraire et centrer
        cropped = inverted[y_min:y_max, x_min:x_max]
        
        # Créer une image 28x28 avec le dessin centré
        img = Image.fromarray(cropped)
        
        # Redimensionner en gardant le ratio
        width, height = img.size
        if width > height:
            new_width = 20
            new_height = int(20 * height / width)
        else:
            new_height = 20
            new_width = int(20 * width / height)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Créer une image 28x28 noire et coller le dessin au centre
        final_img = Image.new('L', (28, 28), 0)
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        final_img.paste(img, (paste_x, paste_y))
        
        # Normaliser et prédire
        img_array = np.array(final_img) / 255.0
        img_flat = img_array.reshape(1, 784)
        
        # Sauvegarder le dessin actuel pour le feedback
        self.current_drawing = img_flat.copy()
        
        # Prédiction
        prediction = self.model.predict(img_flat, verbose=0)
        self.prediction = np.argmax(prediction)
        self.confidence = prediction[0][self.prediction] * 100
        
        # Activer le mode feedback
        self.waiting_feedback = True
    
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
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 50))
        self.screen.blit(title, title_rect)
        
        # === PARTIE GAUCHE : Images d'exemple ===
        if self.sample_images:
            sample_title = FONT_SMALL.render("Exemples d'entraînement (0-9)", True, (148, 163, 184))
            self.screen.blit(sample_title, (50, 120))
            
            # Grille 2x5 d'images (tous les chiffres 0-9)
            img_size = 60
            spacing = 10
            start_x = 50
            start_y = 160
            
            for idx in range(10):
                row = idx // 2
                col = idx % 2
                x = start_x + col * (img_size + spacing)
                y = start_y + row * (img_size + spacing)
                
                # Créer une surface pour l'image
                img_surface = pygame.Surface((img_size, img_size))
                
                # Redimensionner l'image 28x28 vers 60x60
                img_array = self.sample_images[idx]
                img_resized = np.repeat(np.repeat(img_array, img_size//28, axis=0), img_size//28, axis=1)
                
                # Dessiner l'image
                for py in range(img_size):
                    for px in range(img_size):
                        if py < len(img_resized) and px < len(img_resized[0]):
                            gray = img_resized[py, px]
                            img_surface.set_at((px, py), (gray, gray, gray))
                
                self.screen.blit(img_surface, (x, y))
                
                # Label
                label_text = FONT_TINY.render(str(self.sample_labels[idx]), True, SUCCESS_COLOR)
                label_rect = label_text.get_rect(center=(x + img_size // 2, y + img_size + 15))
                self.screen.blit(label_text, label_rect)
            
            # Info sur les exemples sauvegardés
            if self.saved_examples_count > 0:
                saved_info = FONT_TINY.render(f"+ {self.saved_examples_count} exemples des sessions précédentes", True, (148, 163, 184))
                # Décale l'affichage plus bas (ajuste la valeur de y ici)
                self.screen.blit(saved_info, (50, start_y + 400))
        
        # === PARTIE DROITE : Informations d'entraînement ===
        info_x = 350
        
        # Époque
        epoch_text = FONT_MEDIUM.render(f"Époque {self.training_epoch}/5", True, TEXT_COLOR)
        self.screen.blit(epoch_text, (info_x, 120))
        
        # Barre de progression
        bar_width = 550
        bar_height = 35
        bar_x = info_x
        bar_y = 180
        
        pygame.draw.rect(self.screen, PROGRESS_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=20)
        
        if self.training_progress > 0:
            fill_width = int((self.training_progress / 100) * bar_width)
            pygame.draw.rect(self.screen, PROGRESS_FILL, (bar_x, bar_y, fill_width, bar_height), border_radius=20)
        
        # Pourcentage
        percent = FONT_SMALL.render(f"{int(self.training_progress)}%", True, TEXT_COLOR)
        percent_rect = percent.get_rect(center=(bar_x + bar_width // 2, bar_y + bar_height // 2))
        self.screen.blit(percent, percent_rect)
        
        # Métriques
        metrics_y = 250
        if self.training_accuracy > 0:
            acc_text = FONT_SMALL.render(f"Précision entraînement : {self.training_accuracy:.2f}%", True, SUCCESS_COLOR)
            self.screen.blit(acc_text, (info_x, metrics_y))
            
            val_text = FONT_SMALL.render(f"Précision validation : {self.val_accuracy:.2f}%", True, (59, 130, 246))
            self.screen.blit(val_text, (info_x, metrics_y + 35))
            
            loss_text = FONT_SMALL.render(f"Perte : {self.training_loss:.4f}", True, WARNING_COLOR)
            self.screen.blit(loss_text, (info_x, metrics_y + 70))
        
        # === GRAPHIQUES EN TEMPS RÉEL ===
        if len(self.history_accuracy) > 0:
            graph_y = 380
            graph_width = 550
            graph_height = 250
            
            # Fond du graphique
            pygame.draw.rect(self.screen, (30, 41, 59), (info_x, graph_y, graph_width, graph_height), border_radius=10)
            
            # Titre du graphique
            graph_title = FONT_SMALL.render("Évolution de la précision", True, TEXT_COLOR)
            self.screen.blit(graph_title, (info_x + 10, graph_y + 10))
            
            # Dessiner les axes
            axis_margin = 40
            plot_x = info_x + axis_margin
            plot_y = graph_y + 50
            plot_width = graph_width - axis_margin - 20
            plot_height = graph_height - 70
            
            # Axe horizontal et vertical
            pygame.draw.line(self.screen, (100, 116, 139), (plot_x, plot_y + plot_height), 
                           (plot_x + plot_width, plot_y + plot_height), 2)
            pygame.draw.line(self.screen, (100, 116, 139), (plot_x, plot_y), 
                           (plot_x, plot_y + plot_height), 2)
            
            # Labels des axes
            label_0 = FONT_TINY.render("0%", True, (148, 163, 184))
            self.screen.blit(label_0, (plot_x - 30, plot_y + plot_height - 8))
            label_100 = FONT_TINY.render("100%", True, (148, 163, 184))
            self.screen.blit(label_100, (plot_x - 35, plot_y - 8))
            
            # Dessiner les courbes
            if len(self.history_accuracy) > 1:
                # Courbe d'entraînement (vert)
                points_train = []
                for i, acc in enumerate(self.history_accuracy):
                    x = plot_x + (i / (5 - 1)) * plot_width
                    y = plot_y + plot_height - (acc / 100) * plot_height
                    points_train.append((x, y))
                
                if len(points_train) >= 2:
                    pygame.draw.lines(self.screen, SUCCESS_COLOR, False, points_train, 3)
                
                # Courbe de validation (bleu)
                points_val = []
                for i, acc in enumerate(self.history_val_accuracy):
                    x = plot_x + (i / (5 - 1)) * plot_width
                    y = plot_y + plot_height - (acc / 100) * plot_height
                    points_val.append((x, y))
                
                if len(points_val) >= 2:
                    pygame.draw.lines(self.screen, (59, 130, 246), False, points_val, 3)
            
            # Légende
            legend_y = graph_y + graph_height - 25
            pygame.draw.line(self.screen, SUCCESS_COLOR, (info_x + 20, legend_y), (info_x + 50, legend_y), 3)
            legend_train = FONT_TINY.render("Entraînement", True, TEXT_COLOR)
            self.screen.blit(legend_train, (info_x + 55, legend_y - 8))
            
            pygame.draw.line(self.screen, (59, 130, 246), (info_x + 170, legend_y), (info_x + 200, legend_y), 3)
            legend_val = FONT_TINY.render("Validation", True, TEXT_COLOR)
            self.screen.blit(legend_val, (info_x + 205, legend_y - 8))
    
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
        
        # Boutons de base
        self.predict_button.draw(self.screen)
        self.clear_button.draw(self.screen)
        
        # Compteur d'exemples appris
        count_y = 630
        if self.learning_count > 0:
            count_text = FONT_TINY.render(f"📚 {self.learning_count} exemples appris", True, SUCCESS_COLOR)
            self.screen.blit(count_text, (50, count_y))
            
            # Afficher le nombre d'exemples chargés si applicable
            if self.saved_examples_count > 0:
                saved_text = FONT_TINY.render(f"({self.saved_examples_count} depuis sessions précédentes)", True, (148, 163, 184))
                self.screen.blit(saved_text, (50, count_y + 20))
        
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
            
            # Boutons de feedback
            if self.waiting_feedback:
                # Titre feedback
                feedback_title = FONT_SMALL.render("Est-ce correct ?", True, TEXT_COLOR)
                self.screen.blit(feedback_title, (pred_x, 350))
                
                # Boutons Correct / Mauvais
                self.correct_button.draw(self.screen)
                self.wrong_button.draw(self.screen)
                
                # Si on a cliqué sur "Mauvais", afficher les chiffres 0-9
                if hasattr(self, 'show_digit_selection') and self.show_digit_selection:                    
                    for btn in self.digit_buttons:
                        btn.draw(self.screen)
        else:
            # Message
            msg = FONT_SMALL.render("Dessinez puis cliquez 'Prédire'", True, (148, 163, 184))
            self.screen.blit(msg, (pred_x, pred_y + 100))
        
        # Instructions
        inst_y = 660
        if not self.waiting_feedback:
            inst = FONT_TINY.render("Maintenez le bouton gauche de la souris pour dessiner", True, (148, 163, 184))
        else:
            inst = FONT_TINY.render("Donnez votre feedback pour améliorer le modèle !", True, (148, 163, 184))
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
                        # Boutons de feedback
                        if self.waiting_feedback:
                            if self.correct_button.is_clicked(mouse_pos):
                                self.mark_correct()
                                self.clear_canvas()
                                self.show_digit_selection = False
                            elif self.wrong_button.is_clicked(mouse_pos):
                                # Afficher les boutons de sélection
                                self.show_digit_selection = True
                            elif self.show_digit_selection:
                                # Vérifier les boutons de chiffres
                                for i, btn in enumerate(self.digit_buttons):
                                    if btn.is_clicked(mouse_pos):
                                        self.mark_wrong(i)
                                        self.clear_canvas()
                                        self.show_digit_selection = False
                                        break
                        else:
                            # Boutons normaux
                            if self.predict_button.is_clicked(mouse_pos):
                                self.predict_digit()
                            elif self.clear_button.is_clicked(mouse_pos):
                                self.clear_canvas()
                                self.show_digit_selection = False
                            else:
                                self.drawing = True
                                self.draw_on_canvas(mouse_pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.drawing = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.state == "ready":
                    if self.drawing and not self.waiting_feedback:
                        self.draw_on_canvas(mouse_pos)
                    
                    # Hover des boutons
                    if not self.waiting_feedback:
                        self.predict_button.update_hover(mouse_pos)
                        self.clear_button.update_hover(mouse_pos)
                    else:
                        self.correct_button.update_hover(mouse_pos)
                        self.wrong_button.update_hover(mouse_pos)
                        if self.show_digit_selection:
                            for btn in self.digit_buttons:
                                btn.update_hover(mouse_pos)
                
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