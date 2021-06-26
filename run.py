import calorie as Calorie
import train as training
import testfile as te
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Testing project')
    parser.add_argument('--image_path', help="Enter image full path.",default=None)
    parser.add_argument('--train_path', help="Enter the dataset train path.", default="/Users/adityarajaramhegde/Desktop/fyp/Dataset/test")
    parser.add_argument('--test_path', help="Enter dataset test path.", default=None)
    parser.add_argument('--test_model', help="Test the model with all test images", action='store_true')
    parser.add_argument('--segment_dir', help="Path to store segmented images", default=None)
    parser.add_argument('--train_model', help="To train the model", action='store_true')
    parser.add_argument('--no_calorie', help="Don't want to get the calorie of the image", action='store_true')
    parser.add_argument('--epochs', type=int, help="Specify the number of epochs to run", default=60)
    parser.add_argument('--batch_size', type=int, help="Mention number of batch size", default=129)
    parser.add_argument('--learning_rate', type=int, help="Specify learning rate", default=0.001)
    
    args = parser.parse_args()
    image_path = args.image_path
    train_path = args.train_path
    test_path = args.test_path
    train_model = args.train_model
    test_model = args.test_model
    no_calorie = args.no_calorie
    segment_dir = args.segment_dir
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    if train_path:
        if training.train(train_path, epochs, batch_size, learning_rate):
            print("Done with model training. It's time to test the model")
    elif test_path and test_model:
        training.test_model(test_path)
    elif image_path and segment_dir:
        if no_calorie:
            print(training.test(image_path))
        else:
            label =(training.test(image_path))
            print(label)
            print(Calorie.calories(label,image_path, segment_dir))
    else:
        print("Invalid choice. Help:./run.py --help or python3 run.py --help")



