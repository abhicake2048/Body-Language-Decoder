import cv2
import csv
from modules import data_generation as dg
from modules import training as tra
import pickle
from modules import detection as dt



# The termination key for every webcam window is "x"

# Creating csv file for data gathering
filename = 'coords.csv'
with open(filename,mode='w',newline='') as f:
    csv_writer = csv.writer(f,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(dg.get_columns(0))

# Gathering data through the webcam for different feelings
class_names = ["Happy","Sad","Victorious"]
for name in class_names:
    dg.detect_holistic_pose(name,filename,0)


# Initialize the model trainer
model_trainer = tra.BodyLanguageModel()
model = model_trainer.train_model()
model_trainer.save_model(model)

# Train the model and save it to a pickle file. File name = body_language.pkl

with open("body_language.pkl","rb") as f:
    model = pickle.load(f)

dt.make_predictions(model)
