from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.utils import plot_model

left_branch_input = Input(shape=(2,), name='Left_input')
left_branch_output = Dense(5, activation='relu')(left_branch_input)

right_branch_input = Input(shape=(2,), name='Right_input')
right_branch_output = Dense(5, activation='relu')(right_branch_input)

concat = concatenate([left_branch_output, right_branch_output], name='Concatenate')
final_model_output = Dense(3, activation='sigmoid')(concat)
final_model = Model(inputs=[left_branch_input, right_branch_input], outputs=final_model_output,
                    name='Final_output')
final_model.compile(optimizer='adam', loss='binary_crossentropy')
# To train
# final_model.fit([Left_data,Right_data], labels, epochs=10, batch_size=32)