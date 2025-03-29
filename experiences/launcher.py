import subprocess
from notify import notify_discord

try:
    print("🚀 Entraînement de experience2...")
    notify_discord(f"🚀 Entraînement de experience2")

    subprocess.run(["python", "experience_02.py"], check=True)
    print("✅ experience2 terminé avec succès.\n")

    print("🚀 Entraînement de experience3...")
    notify_discord(f"🚀 Entraînement de experience3")

    subprocess.run(["python", "experience_03.py"], check=True)
    print("✅ experience3 terminé avec succès.\n")

    print("🚀 Entraînement de experience4...")
    notify_discord(f"🚀 Entraînement de experience4")

    subprocess.run(["python", "experience_04.py"], check=True)
    print("✅ experience4 terminé avec succès.\n")

except subprocess.CalledProcessError as e:
    print("❌ Une erreur est survenue lors de l'exécution :")
    notify_discord(f"❌ Une erreur est survenue lors de l'exécution : {e}")
    print(e)