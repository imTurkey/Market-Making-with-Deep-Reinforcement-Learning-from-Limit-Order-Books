# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from tensorflow import keras
from keras import backend as K

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True                                #按需分配显存
K.set_session(tf.compat.v1.Session(config=config))

def get_lob_model(latent_dim, T):
    lob_state = keras.layers.Input(shape=(T, 40, 1))

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(lob_state)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 5), strides=(1, 5))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 4))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = keras.layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    attn_input = conv_reshape
    attn_input_last = attn_input[:,-1:,:]  

    multi_head_attn_layer_1 = keras.layers.MultiHeadAttention(num_heads=10, key_dim=16, output_shape=64)

    attn_output, weight = multi_head_attn_layer_1(attn_input_last, attn_input, return_attention_scores=True)

    attn_output = keras.layers.Flatten()(attn_output)

    # add Batch Normalization
    # attn_output = keras.layers.BatchNormalization()(attn_output)

    # add Layer Normalization
    # attn_output = keras.layers.LayerNormalization()(attn_output)
    
    return keras.models.Model(lob_state, attn_output)


def get_fclob_model(latent_dim,T):
    print("This is the FC-LOB model")
    lob_state = keras.layers.Input(shape=(T, 40, 1))

    dense_input = keras.layers.Flatten()(lob_state)

    dense_output = keras.layers.Dense(1024, activation='leaky_relu')(dense_input)
    dense_output = keras.layers.Dense(256, activation='leaky_relu')(dense_input)
    dense_output = keras.layers.Dense(latent_dim, activation='leaky_relu')(dense_input)

    return keras.models.Model(lob_state, dense_output)

def compute_output_shape(input_shape):
    return (input_shape[0], 64)

def get_pretrain_model(model, T):
    lob_state = keras.layers.Input(shape=(T, 40, 1))
    embedding = model(lob_state)
    output = keras.layers.Dense(3, activation='softmax')(embedding)
    
    return keras.models.Model(lob_state, output)

def get_model(lob_model, T, with_lob_state=True, with_market_state=True, with_agent_state=True):
    input_ls = list()
    dense_input = list()
    if with_lob_state:
        lob_state = keras.layers.Input(shape=(T, 40, 1))
        encoder_outputs = lob_model(lob_state) 
        input_ls.append(lob_state)
        dense_input.append(encoder_outputs)
    else:
        print('w/o lob state!')  

    if with_agent_state:
        agent_state = keras.layers.Input(shape=(24,))
        input_ls.append(agent_state)
        dense_input.append(agent_state)
    else:
         print('w/o agent state!')  
    
    if with_market_state:
        market_state = keras.layers.Input(shape=(24,))
        input_ls.append(market_state)
        dense_input.append(agent_state)
    else:
        print('w/o market state!')  

    dense_input = keras.layers.concatenate(dense_input, axis=1)

    dense_output = keras.layers.Dense(64, activation='leaky_relu')(dense_input)

    return keras.models.Model(input_ls, dense_output)

if __name__ == '__main__':
    get_lob_model(64,50).summary()