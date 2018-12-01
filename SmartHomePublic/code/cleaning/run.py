import os

if not os.path.exists("../../temp"):
    os.mkdir("../../temp")

if not os.path.exists("../../temp"):
    os.mkdir("../../output")

os.system("python preclean.py")
os.system("python build.py")
