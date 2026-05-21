import re

with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# REMOVE URLs
text = re.sub(r"http\S+", "", text)

# REMOVE EXTRA SPACES
text = re.sub(r"\s+", " ", text)

# REMOVE WEBSITE JUNK
bad_patterns = [
    "Skip to Content",
    "Products and Plans",
    "Privacy Manager",
    "Task Tracker",
    "Accessibility",
    "Request a Demo",
    "Subscription",
    "My Cart",
    "Teacher Accounts",
    "Pricing For Schools",
    "Go ad-free",
    "Tired of Ads",
    "What's NEW at TPC",
    "Return to Screen Reader Navigation",
    "Physics Classroom",
    "Concept Builders",
    "Calc Pad",
    "Minds On",
    "NGSS",
    "Teacher Aids",
    "Lesson Plans",
    "Solutions Guide",
    "Student Extras",
    "Webinars and Trainings",
    "Edit Profile Settings",
    "Quote or Order"
]

for pattern in bad_patterns:
    text = text.replace(pattern, "")

# REMOVE REPEATED WORD BLOCKS
text = re.sub(r"(Physics Tutorial\s*){2,}", "", text)
text = re.sub(r"(Newton's Laws\s*){3,}", "Newton's Laws ", text)

# KEEP ONLY IMPORTANT LESSON CONTENT
important_sections = []

keywords = [
    "Lesson",
    "Newton",
    "Force",
    "Velocity",
    "Acceleration",
    "Momentum",
    "Energy",
    "Projectile",
    "Vectors",
    "Kinematics",
    "Motion"
]

sentences = text.split(".")

for s in sentences:
    for k in keywords:
        if k.lower() in s.lower():
            important_sections.append(s)
            break

clean_text = ". ".join(important_sections)

with open("clean_notes.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("Clean notes created successfully!")