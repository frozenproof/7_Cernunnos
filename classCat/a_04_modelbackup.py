import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout

class MyClassModel(Model):
    def __init__(self, num_classes):
        super(MyClassModel, self).__init__()
        self.num_classes = num_classes
        self.layers_list = []

    def add_layer(self, units, activation='relu', dropout_rate=0.0, input_shape=None):
        if not self.layers_list:
            # If it's the first layer, it needs input shape
            if input_shape is None:
                raise ValueError("Input shape must be provided for the first layer.")
            self.layers_list.append(Dense(units, activation=activation, input_shape=input_shape))
        else:
            self.layers_list.append(Dense(units, activation=activation))
        
        if dropout_rate > 0:
            self.layers_list.append(Dropout(dropout_rate))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

    def build_model(self, input_shape):
        super().build(input_shape)
        # Set built flag to True
        self.built = True



class BackupModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, backup_frequency=5, monitor='val_loss', mode='min', **kwargs):
        """
        Custom callback to save model weights with a specified backup frequency
        and to save only the best model based on a monitored metric.

        Args:
            *args: Variable length argument list.
            backup_frequency (int): Frequency of backups (default is 5 epochs).
            monitor (str): Metric to monitor for determining the best model (default is 'val_loss').
            mode (str): One of {'auto', 'min', 'max'}. Mode to determine the best model.
                If set to 'min', the callback will save the model when the monitored quantity is minimized.
                If set to 'max', it will save the model when the monitored quantity is maximized.
                If set to 'auto', the mode is inferred based on the monitored quantity.
            **kwargs: Additional keyword arguments for tf.keras.callbacks.ModelCheckpoint.
        """
        super().__init__(*args, **kwargs)
        self.backup_frequency = backup_frequency
        self.monitor = monitor
        self.mode = mode
        # Initialize best value as infinity for minimization and negative infinity for maximization
        # Specify the directory to save the checkpoints
        checkpoint_dir = 'bestcheckpoints/'

        # Create the directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        """
        Called by Keras at the end of every epoch. Saves the model weights with a specified backup frequency
        and saves only the best model based on the monitored metric.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of logs containing the loss value and all the metrics
                at the end of the current epoch.
        """
        # Check if the current epoch number satisfies the backup frequency condition
        if (epoch + 1) % self.backup_frequency == 0:
            # Construct the filepath for saving the model with backup frequency
            filepath = "checkpoints/"+self.filepath.format(epoch=epoch + 1, **logs)+str(epoch)
            self.model.save_weights(filepath, overwrite=True)
            print("\nSaved model checkpoint at epoch", epoch + 1, "\n")

        # Retrieve the current value of the monitored metric
        loss = logs['loss']
        accuracy = logs['accuracy']
        val_loss = logs['val_loss']
        val_accuracy = logs['val_accuracy']

        # Convert the values to strings
        loss_str = str(round(loss,5))
        accuracy_str = str(round(accuracy,5))
        val_loss_str = str(round(val_loss,5))
        val_accuracy_str = str(round(val_accuracy,5))
        if loss is None:
            print(f"WARNING: Could not find metric '{self.monitor}' in logs. Model checkpoint not saved.")
            return
        # Construct the checkpoint filename using the extracted values
        checkpoint_name = f"ckpt_val_loss{val_loss_str}"

        print(checkpoint_name+f"_loss{loss_str}_accuracy{accuracy_str}_val_accuracy{val_accuracy_str}\n")

        # Determine whether the current value represents an improvement based on the mode
        if self.mode == 'min':
            improvement = val_loss < self.best_value
        else:
            improvement = val_loss > self.best_value

        # If there is an improvement, save the model weights and update the best value
        if improvement:
            self.best_value = loss
            filepath = "bestcheckpoints/"+checkpoint_name
            self.model.save_weights(filepath, overwrite=True)
            print("\nSaved best model checkpoint at epoch", epoch + 1, "\n")


# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu")) 
# model.add(Dropout (0.5))
# model.add(Dense (64, activation="relu"))
# model.add(Dropout (0.5))
# model.add(Dense(len(train_Y[0]), activation="softmax"))
# adam=tf.keras.optimizers.Adam (learning_rate=0.01) 
# model.compile(loss="categorical_crossentropy",optimizer=adam,metrics=["accuracy"])        