from inference_helpers import *

if __name__ == '__main__':
    # get file names in data directory
    filenames = get_filenames()
    # https://rock-it.pl/how-to-reuse-keras-deep-neural-network/
    # load model
    model = None
    model = load_model("model.h5", custom_objects={'f1':f1})
    #model.summary()
    
    # inference
    for path in filenames:
        img = None
        img = get_data(path)
        pred, pred_arg = inference(img, model)
        image_class = decode(pred_arg)
        print("{:30}   {}".format(path, image_class))