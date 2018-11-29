import os

if os.path.exists("../../temp"):
    os.system("rm -rf ../../temp")

if os.path.exists("../../output"):
    os.system("rm -rf ../../output")

os.mkdir("../../temp")
os.mkdir("../../output")

os.system("python preclean.py")
os.system("python build.py")
