# Usage
# python main.py --dataset "dataset folder"

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from bis29_.datasets.datasetloader import SimpleDatasetLoader
from bis29_.preprocessing.preprocessor import SimplePreprocessor
from imutils import paths
import argparse

# construct an argparse to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, type=str, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="number of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="number of jobs for KNN distance")
args = vars(ap.parse_args())

# grab the list of images that will be describing
print("[INFO] loading images")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
sp = SimplePreprocessor(width=32, height=32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths=imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of the data for training
# and the remaining for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier.....")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))