import os
rootrecord = []
dirrecord = []
for root,dirs,files in os.walk("CT-Lung/"):
    if len(dirs) == 0:
        rootrecord.append(root)
with open("filedir.txt",'w') as f:
    f.write('\n'.join(rootrecord))
