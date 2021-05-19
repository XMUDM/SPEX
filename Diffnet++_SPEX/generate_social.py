import pickle
with open("data/twitter/rec/friends.dic","rb") as f:
    fs = pickle.load(f)
with open("data/twitter/rec/social.share","w") as f:
    for u in fs:
        for i in fs[u]:
            f.write(u+" "+i+"\n")
