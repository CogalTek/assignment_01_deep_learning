import requests

webhook_url = "https://discord.com/api/webhooks/1354904572589641859/fe-4khag8nBwRIpBTbbToNF4Wal0w0LdGik-ZKpxFKUK3O5BHe-_SjYeB_KF6GbKufxy"
user = "<@269964207363981313>"

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