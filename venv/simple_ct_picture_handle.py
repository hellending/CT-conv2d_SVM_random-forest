from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=128,height=128):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("C:\\Users\\zkzs5\\Desktop\\data\\CT_NonCOVID\\*.jpg"):
    convertjpg(jpgfile,"C:\\Users\\zkzs5\\Desktop\\non")