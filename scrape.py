import requests
from bs4 import BeautifulSoup

# Read URLs from notes.txt
with open("notes.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

urls = []

for line in lines:
    if "http" in line:
        parts = line.split("http")
        url = "http" + parts[1].strip()
        urls.append(url)

all_text = ""

for url in urls:

    print(f"Scraping: {url}")

    try:
        response = requests.get(url)

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove useless elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        # Clean spaces
        text = " ".join(text.split())

        all_text += text + "\n\n"

    except Exception as e:
        print(f"Failed: {url}")
        print(e)

# Save cleaned notes
with open("clean_notes.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("\nDONE! Clean notes saved.")