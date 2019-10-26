import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings


from tensorflow.keras.layers import Conv2D, MaxPool2D, ReLU, BatchNormalization

def cnn(model):
    '''
    Convolutional Neural Network feature extractor backbone
    adapted from alexnet model

    # Arguments

    model:      A keras functional model to add the cnn layers onto

    # modifies

    The given model by adding the cnn layers
    '''
    model = Conv2D(64, kernel_size=3, padding='same')(model)
    #model = BatchNormalization()(model)
    model = ReLU()(model)
    model = MaxPool2D(pool_size=(2, 2))(model) # divides by 2
    model = Conv2D(128, kernel_size=3, padding='same')(model)
    #model = BatchNormalization()(model)
    model = ReLU()(model)
    model = MaxPool2D(pool_size=(2, 2))(model) # divides by 2
    model = Conv2D(256, kernel_size=3, padding='same')(model)
    #model = BatchNormalization()(model)
    model = ReLU()(model)
    model = MaxPool2D(pool_size=(2, 4))(model) # divides by 4
    model = Conv2D(512, kernel_size=3, padding='same')(model)
    #model = BatchNormalization()(model)
    model = ReLU()(model)
    model = MaxPool2D(pool_size=(1, 4))(model) # divides w by 4
    model = Conv2D(512, kernel_size=3, padding='same')(model)
    #model = BatchNormalization()(model)
    model = ReLU()(model)
    model = MaxPool2D(pool_size=(4, 4))(model) # divides by 4
    model = Conv2D(1024, kernel_size=3, padding='same')(model)
    #model = BatchNormalization()(model)
    model = ReLU()(model)
    return model