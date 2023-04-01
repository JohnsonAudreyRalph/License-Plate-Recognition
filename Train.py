from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = 'Data'
train_generator = train_datagen.flow_from_directory(
    path+'/train',
    target_size=(28,28),
    batch_size=1,
    class_mode='sparse'
)

validation_generator = train_datagen.flow_from_directory(
    path+'/val',
    target_size=(28,28),
    class_mode='sparse'
)

def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')

def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)

K.clear_session()
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.0001), metrics=[custom_f1score])
model.summary()

class stop_training_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if(logs.get('val_custom_f1score') > 0.99):
            self.model.stop_training = True

batch_size = 1
callbacks = [stop_training_callback()]
history = model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, validation_data=validation_generator, epochs=200, verbose=1, callbacks=callbacks)
model.save('Data/Save_mode.hdf5')

plt.figure(1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Biểu đồ trực quan hóa sự mất mát trong quá trình huấn luyện')
plt.legend()
plt.show()
# plt.savefig('Visualization_Chart_of_Loss.png')

plt.figure(1)
plt.plot(history.history['custom_f1score'], label='custom_f1score')
plt.plot(history.history['val_custom_f1score'], label='val_custom_f1score')
plt.title('Biểu đồ trực quan hóa đánh giá độ chính xác của quá trình huấn luyện')
plt.legend()
plt.show()
# plt.savefig('Visualization_Chart_of_Accuracy.png')