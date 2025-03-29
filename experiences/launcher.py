import subprocess
from notify import notify_discord

try:
    # print("🚀 Entraînement de experience_01...")
    # subprocess.run(["python", "experience_01.py"], check=True)
    # print("✅ experience_01 terminé avec succès.\n")

    # print("🚀 Entraînement de stanford_train...")
    # subprocess.run(["python", "stanford_train.py"], check=True)
    # print("✅ stanford_train terminé avec succès.\n")

    print("🚀 Entraînement de experience2...")
    notify_discord(f"🚀 Entraînement de experience2")

    subprocess.run(["python", "experience_02.py"], check=True)
    print("✅ experience2 terminé avec succès.\n")

    print("🚀 Entraînement de experience3...")
    notify_discord(f"🚀 Entraînement de experience3")

    subprocess.run(["python", "experience_03.py"], check=True)
    print("✅ experience3 terminé avec succès.\n")

except subprocess.CalledProcessError as e:
    print("❌ Une erreur est survenue lors de l'exécution :")
    notify_discord(f"❌ Une erreur est survenue lors de l'exécution : {e}")
    print(e)