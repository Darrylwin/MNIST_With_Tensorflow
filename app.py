"""
DÉMO MNIST - Version Simplifiée pour Présentation
Groupe 11 - Big Data - Deep Learning
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Désactiver les warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def afficher_titre(texte):
    """Affiche un titre formaté"""
    print("\n" + "="*70)
    print(f"  {texte}")
    print("="*70)

def afficher_section(texte):
    """Affiche une section"""
    print(f"\n🔹 {texte}")

# ============================================================================
# DÉBUT DE LA DÉMO
# ============================================================================

afficher_titre("DÉMO MNIST - RECONNAISSANCE DE CHIFFRES MANUSCRITS")
print("\n👥 Groupe 11 : DEGBEY, DOSSOU, DOUFFAN, LOGOSSOU")
print("📚 Sujet : Deep Learning et Réseaux Neuronaux")

input("\n▶️  Appuyez sur ENTRÉE pour commencer la démo...")

# ============================================================================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ============================================================================

afficher_section("ÉTAPE 1 : Chargement du dataset MNIST")
print("   Le dataset MNIST contient 70,000 images de chiffres manuscrits (0-9)")

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"\n   ✅ Données chargées !")
print(f"   📊 {x_train.shape[0]:,} images pour l'entraînement")
print(f"   📊 {x_test.shape[0]:,} images pour le test")
print(f"   📏 Taille de chaque image : {x_train.shape[1]}x{x_train.shape[2]} pixels")

input("\n▶️  Appuyez sur ENTRÉE pour voir quelques exemples d'images...")

# Afficher quelques exemples
plt.figure(figsize=(12, 4))
plt.suptitle("Exemples d'images du dataset MNIST", fontsize=14, fontweight='bold')
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Chiffre : {y_train[i]}", fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.savefig('1_exemples_mnist.png', dpi=150)
print("\n   💾 Image sauvegardée : 1_exemples_mnist.png")
plt.show(block=False)
plt.pause(2)

input("\n▶️  Appuyez sur ENTRÉE pour passer au prétraitement...")

# ============================================================================
# ÉTAPE 2 : PRÉTRAITEMENT
# ============================================================================

afficher_section("ÉTAPE 2 : Prétraitement des données")
print("   Transformation nécessaire avant l'entraînement :")
print("   1️⃣  Normalisation : pixels 0-255 → 0-1")
print("   2️⃣  Aplatissement : images 28x28 → vecteurs de 784")

# Normalisation
x_train = x_train / 255.0
x_test = x_test / 255.0

# Aplatissement
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

print(f"\n   ✅ Prétraitement terminé !")
print(f"   📐 Nouvelle forme des données : {x_train_flat.shape}")

input("\n▶️  Appuyez sur ENTRÉE pour construire le réseau de neurones...")

# ============================================================================
# ÉTAPE 3 : CONSTRUCTION DU MODÈLE
# ============================================================================

afficher_section("ÉTAPE 3 : Construction du réseau de neurones")
print("   Architecture du modèle :")
print("   🔴 Couche d'entrée : 784 neurones (28x28 pixels)")
print("   🟠 Couche cachée 1 : 128 neurones + ReLU")
print("   🟡 Dropout : 20% (évite le surapprentissage)")
print("   🟢 Couche cachée 2 : 64 neurones + ReLU")
print("   🔵 Couche de sortie : 10 neurones + Softmax (chiffres 0-9)")

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n   ✅ Modèle créé et compilé !")
print(f"   🧮 Paramètres à entraîner : {model.count_params():,}")

input("\n▶️  Appuyez sur ENTRÉE pour lancer l'entraînement...")

# ============================================================================
# ÉTAPE 4 : ENTRAÎNEMENT
# ============================================================================

afficher_section("ÉTAPE 4 : Entraînement du modèle")
print("   ⏱️  Cela prendra environ 1-2 minutes...")
print("   📈 Suivez l'évolution de la précision (accuracy)\n")

history = model.fit(
    x_train_flat, 
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

print("\n   ✅ Entraînement terminé !")

input("\n▶️  Appuyez sur ENTRÉE pour évaluer les performances...")

# ============================================================================
# ÉTAPE 5 : ÉVALUATION
# ============================================================================

afficher_section("ÉTAPE 5 : Évaluation sur les données de test")

test_loss, test_accuracy = model.evaluate(x_test_flat, y_test, verbose=0)

print(f"\n   🎯 RÉSULTATS FINAUX :")
print(f"   {'─'*50}")
print(f"   Précision (Accuracy) : {test_accuracy*100:.2f}%")
print(f"   Perte (Loss)         : {test_loss:.4f}")
print(f"   {'─'*50}")

if test_accuracy > 0.97:
    print("   🏆 Excellent résultat ! Le modèle est très performant.")
elif test_accuracy > 0.95:
    print("   ✨ Bon résultat ! Le modèle fonctionne bien.")
else:
    print("   ⚠️  Le modèle pourrait être amélioré.")

input("\n▶️  Appuyez sur ENTRÉE pour voir les graphiques...")

# Créer les graphiques
plt.figure(figsize=(14, 5))

# Graphique 1 : Précision
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-', label='Entraînement', linewidth=2)
plt.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
plt.title('Évolution de la Précision', fontsize=14, fontweight='bold')
plt.xlabel('Époque', fontsize=12)
plt.ylabel('Précision (%)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Graphique 2 : Perte
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b-', label='Entraînement', linewidth=2)
plt.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
plt.title('Évolution de la Perte', fontsize=14, fontweight='bold')
plt.xlabel('Époque', fontsize=12)
plt.ylabel('Perte', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('2_courbes_apprentissage.png', dpi=150)
print("\n   💾 Graphiques sauvegardés : 2_courbes_apprentissage.png")
plt.show(block=False)
plt.pause(2)

input("\n▶️  Appuyez sur ENTRÉE pour tester le modèle sur de nouvelles images...")

# ============================================================================
# ÉTAPE 6 : PRÉDICTIONS
# ============================================================================

afficher_section("ÉTAPE 6 : Test de prédiction")
print("   Le modèle va maintenant prédire des chiffres qu'il n'a jamais vus\n")

# Faire 10 prédictions aléatoires
plt.figure(figsize=(15, 6))
plt.suptitle("Prédictions du Modèle sur de Nouvelles Images", fontsize=14, fontweight='bold')

correct = 0
for i in range(10):
    idx = np.random.randint(0, len(x_test))
    image = x_test[idx]
    prediction = model.predict(x_test_flat[idx:idx+1], verbose=0)
    predicted_digit = np.argmax(prediction)
    true_digit = y_test[idx]
    confidence = prediction[0][predicted_digit] * 100
    
    if predicted_digit == true_digit:
        correct += 1
        color = 'green'
        status = "✓"
    else:
        color = 'red'
        status = "✗"
    
    print(f"   {status} Test {i+1:2d} : Prédit = {predicted_digit}, Réel = {true_digit}, Confiance = {confidence:.1f}%")
    
    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f'P:{predicted_digit} | R:{true_digit}\n{confidence:.0f}%', 
              color=color, fontsize=10, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.savefig('3_predictions.png', dpi=150)
print(f"\n   💾 Prédictions sauvegardées : 3_predictions.png")
print(f"   📊 Réussite : {correct}/10 prédictions correctes")
plt.show(block=False)
plt.pause(2)

# ============================================================================
# CONCLUSION
# ============================================================================

afficher_titre("CONCLUSION")
print("""
✅ Ce que nous avons démontré :

1. 📥 Chargement et exploration d'un dataset réel (MNIST)
2. 🔧 Prétraitement des données pour le deep learning
3. 🧠 Construction d'un réseau de neurones avec plusieurs couches
4. 🚀 Entraînement du modèle (apprentissage des patterns)
5. 📊 Évaluation des performances (~98% de précision)
6. 🔮 Utilisation du modèle pour faire des prédictions

🎯 POINTS CLÉS :
   • Un réseau simple avec 3 couches suffit pour atteindre 98% de précision
   • Le modèle apprend automatiquement à reconnaître les chiffres
   • L'entraînement prend quelques minutes sur un PC standard
   • Le deep learning est accessible et pratique !

📚 Ce modèle illustre les concepts fondamentaux du deep learning
   que nous avons présentés dans notre exposé.
""")

afficher_titre("FIN DE LA DÉMO - MERCI !")
print("\n👥 Groupe 11 : DEGBEY, DOSSOU, DOUFFAN, LOGOSSOU")
print("📧 Questions ? N'hésitez pas !\n")

input("▶️  Appuyez sur ENTRÉE pour fermer...")
plt.close('all')