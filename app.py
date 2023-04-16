from keras.models import load_model
from keras.optimizers import  Adam
import cv2
import numpy as np
import utily
import os
from tqdm import tqdm
import json

model = load_model('model.h5',compile=False)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])


def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (utily.IMG_SIZE, utily.IMG_SIZE))
    img = np.array(img).reshape(-1, utily.IMG_SIZE, utily.IMG_SIZE, 3) / 255.0
    result = model.predict(img)
    return utily.map_output[np.argmax(result)]


outputs = {}
test_dir = "test/anonymous"
for image in tqdm(os.listdir(test_dir)):
    full_path = os.path.join(test_dir,image)
    outputs[image] = predict(full_path)

with open("predictions.json", "w") as outfile:
    json.dump(outputs, outfile)


truth = open('ground_truth.json')
truth = json.load(truth)

predictions = open('predictions.json')
predictions = json.load(predictions)

print(predictions['0001.jpg'])
correct = 0

for item in truth.keys():

    if truth[item] == predictions[item]:
        correct += 1

print(f" Accuracy In Testing is {correct / len(truth)}")
