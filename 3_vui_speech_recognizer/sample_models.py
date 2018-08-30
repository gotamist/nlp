from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, Dropout, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, drop_fraction = 0.2):
    """ Build a recurrent + convolutional network for speech 
    """
   
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    bn_cnn = Dropout(drop_fraction)(bn_cnn)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    bn_rnn = Dropout(drop_fraction)(bn_rnn)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    
    # Do this without dropout to study the effect of dropout later
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    
#        simp_rnn = GRU(units, activation=activation,
#        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
#    bn_rnn = BatchNormalization()(simp_rnn)
#    cells = [ GRU(output_dim) for _ in recur_layers ]
#    rnn_layer = RNN(cells)(input_data)
    recur = input_data
    for cellnum in range(recur_layers):
        recur = GRU( units, activation='relu', return_sequences=True, implementation=2, name='rnn_'+str(cellnum))(recur)
        recur = BatchNormalization()(recur)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(recur)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29, drop_fraction=0.0):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional( GRU( units, activation='relu', 
                                   return_sequences=True, implementation=2,dropout=drop_fraction, 
                                   recurrent_dropout=drop_fraction ) )(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deep_bd_rnn_model(input_dim, units, recur_layers=2, output_dim=29, drop_fraction=0.0):
    """ Build a deepbidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    recur = input_data
    for cellnum in range(recur_layers):
        recur = Bidirectional( GRU( units, activation='relu', return_sequences=True, dropout=drop_fraction, recurrent_dropout=drop_fraction) )(recur)
        recur = BatchNormalization()(recur)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))( recur )
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

    
#cnn_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
#deep_bd_rnn_model(input_dim, units, recur_layers=2, output_dim=29):

def opt_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recur_layers=2, output_dim=29):
        #only 1 convolutional layer
    #pool_size=1, 
    """ Build a deep network for speech 
    """
    
#    pool_strides =1
    dilation_rate = 1
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network

    # Add convolutional layer with BN and Dropout
    drop_fraction = 0.2
    conv1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=dilation_rate,
                     name='conv1d_1')(input_data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(drop_fraction)(conv1)
    #In the maxpoling layer below, I have used kernel_size from the conv layer as the pooling size.
#    conv1 = MaxPooling1D(pool_size=kernel_size,strides=pool_strides, padding='same')(conv1) #default padding is 'valid' (strides any less would make the sequence length<222)
    
    
    # Add a second convolutional layer with BN and Dropout
    
#    conv2 = Conv1D(2*filters, kernel_size, 
#                     strides=conv_stride, 
#                     padding=conv_border_mode,
#                     activation='relu',
#                     dilation_rate=dilation_rate,
#                     name='conv1d_2')(conv1)
#    conv2 = BatchNormalization()(conv2)
#    conv2 = Dropout(drop_fraction)(conv2)
    
#    conv2 = MaxPooling1D(pool_size=kernel_size,strides=pool_strides, padding='same')(conv2) #default padding is 'valid' (strides any less would make the sequence length<222)
 
    
    # Add multilayer RNN with dropout
    
    recur = conv1
    for cellnum in range(recur_layers):
        recur = Bidirectional( GRU( units, activation='relu', return_sequences=True, dropout = drop_fraction, recurrent_dropout = drop_fraction) )(recur) #dropout was droput_U and recurrent_dropout was dropout_W
        recur = BatchNormalization()(recur)
        
#    recur = MaxPooling1D( pool_size=kernel_size, strides=pool_strides, padding='same' )( recur ) #default padding is 'valid' (strides any less would make the sequence length<222)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))( recur )
#    time_dense = Dropout(drop_fraction)(time_dense)
    
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
#    model.output_length = lambda x: cnn_pooling_output_length(
#        x, kernel_size, conv_border_mode, conv_stride, pool_size=pool_size, num_cnn=2, num_pool=1)

#    model.output_length = lambda x: cnn_output_length( cnn_output_length(
#        x, kernel_size, conv_border_mode, conv_stride, dilation=dilation_rate), kernel_size, conv_border_mode, conv_stride, dilation=dilation_rate )
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, recur_layers=2, output_dim=29): 
#    two convolutional layers
    """ Build a deep network for speech 
    """
    
#    pool_strides =1
    dilation_rate = 1
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network

    # Add convolutional layer with BN and Dropout
    drop_fraction = 0.2
    conv1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=dilation_rate,
                     name='conv1d_1')(input_data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(drop_fraction)(conv1)
    #In the maxpoling layer below, I have used kernel_size from the conv layer as the pooling size.
#    conv1 = MaxPooling1D(pool_size=kernel_size,strides=pool_strides, padding='same')(conv1) #default padding is 'valid' (strides any less would make the sequence length<222)
#    Add a second convolutional layer with BN and Dropout
    
    conv2 = Conv1D(2*filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=dilation_rate,
                     name='conv1d_2')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(drop_fraction)(conv2)
    
#   conv2 = MaxPooling1D(pool_size=kernel_size,strides=pool_strides, padding='same')(conv2) #default padding is 'valid' (strides any less would make the sequence length<222)
    # Add multilayer RNN with dropout
    
    recur = conv2
    for cellnum in range(recur_layers):
        recur = Bidirectional( GRU( units, activation='relu', return_sequences=True, dropout = drop_fraction, recurrent_dropout = drop_fraction) )(recur) #dropout was droput_U and recurrent_dropout was dropout_W
        recur = BatchNormalization()(recur)
        
#    recur = MaxPooling1D( pool_size=kernel_size, strides=pool_strides, padding='same' )( recur ) #default padding is 'valid' (strides any less would make the sequence length<222)

    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))( recur )
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
#    model.output_length = lambda x: cnn_output_length(
#        x, kernel_size, conv_border_mode, conv_stride)
#    model.output_length = lambda x: cnn_pooling_output_length(
#        x, kernel_size, conv_border_mode, conv_stride, pool_size=pool_size, num_cnn=2, num_pool=1)

    model.output_length = lambda x: cnn_output_length( cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation=dilation_rate), kernel_size, conv_border_mode, conv_stride, dilation=dilation_rate )
    print(model.summary())
    return model