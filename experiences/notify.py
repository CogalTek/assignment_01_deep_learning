import requests

webhook_url = "url de ton bot"
user = "<@id_user>" # A changer

def notify_discord(message, mention = True):
    if mention:
        message = user + message
    data = {
        "content": message
    }
    response = requests.post(webhook_url, json=data)
    if response.status_code == 204:
        print("✅ Notification Discord envoyée avec succès.")
    else:
        print("❌ Erreur:", response.text)