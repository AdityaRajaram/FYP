import calorie as Calorie
import train as training
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Testing project')
    parser.add_argument('--image_path', help="Enter image full path.", default = None)
    parser.add_argument('--train_path', help="Enter the dataset train path.", default =None)
    parser.add_argument('--test_path', help="Enter dataset test path.", default =None)
    parser.add_argument('--test_model', help="Test the model with all test images", action='store_true')
    parser.add_argument('--train_model', help="To train the model", action='store_true')
    parser.add_argument('--no_calorie', help="Don't want to get the calorie of the image", action='store_true')
    args = parser.parse_args()
    image_path = args.image_path
    train_path = args.train_path
    test_path = args.test_path
    train_model = args.train_model
    test_model = args.test_model
    no_calorie = args.no_calorie
    print(args)
    if train_path and train_model:
        if training.train(train_path):
            print("Done with model training. It's time to test the model")
    elif test_path and test_model:
        folders = os.listdir(test_path)
        training.test_model(test_path)
    elif image_path:
        if no_calorie:
            label = training.test(image_path)
            Calorie.calories(label,image_path)
        else:
            print(training.test(image_path))
    else:
        print("Invalid choice. Help:./run.py --help or python3 run.py --help")



