import subprocess

try:
    print("🚀 Entraînement de experience_01...")
    subprocess.run(["python", "experience_01.py"], check=True)
    print("✅ experience_01 terminé avec succès.\n")

    print("🚀 Entraînement de stanford_train...")
    subprocess.run(["python", "stanford_train.py"], check=True)
    print("✅ stanford_train terminé avec succès.\n")

except subprocess.CalledProcessError as e:
    print("❌ Une erreur est survenue lors de l'exécution :")
    print(e)