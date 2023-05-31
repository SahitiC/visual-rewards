import os
import pandas as pd

#%%

folder = "../stimuli/resized_images"
file_list = []

for count, foldername in enumerate(os.listdir(folder)):
    print(foldername)
    for count2, filename in enumerate(os.listdir(folder+"/"+foldername)):
        
        dst = f"{foldername}{str(count2)}.jpg"
        src =f"{folder}/{foldername}/{filename}"  # foldername/filename, if .py file is outside folder
        #dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
    
#%%

folder = "../stimuli/matched_images"
file_list = []
   
for count, filename in enumerate(os.listdir(folder)):

    file_list = file_list + [folder+"/"+filename]
  
stimuli = pd.DataFrame(file_list, columns = ['stimuli'])
stimuli.to_csv('stimuli.csv', index=False)