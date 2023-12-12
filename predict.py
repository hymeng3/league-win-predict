import sys
import argparse

import utils
import models



parser = argparse.ArgumentParser(description="A win probability predictor for League of Legends",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("input", help="Path to input CSV file with game states")
parser.add_argument("-t", "--train", action="store_true", 
                    help="Uses input to train models instead")

args = parser.parse_args()

input_file = args.input

models = models.Models()

if args.train:
    
    train_set, train_labels, valid_set, valid_labels = utils.process_input(input_file)

    models.train(train_set, train_labels)
    models.evaluate(valid_set, valid_labels)

else:
    input_df = utils.read_input(input_file)
    data = utils.feature_transform(input_df)
    scaled_data = utils.scale_dataset(data)


    predictions = models.predict(scaled_data)

    print(predictions)
