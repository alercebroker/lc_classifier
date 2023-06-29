import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import precision_recall_fscore_support


class MLP(Model):
    def __init__(self, list_of_classes):
        super().__init__()
        self.list_of_classes = list_of_classes
        self.dense_layer_1 = Dense(
            300,
            activation='relu',
            kernel_initializer='he_uniform'
        )
        self.dropout_layer = Dropout(0.5)
        self.dense_layer_2 = Dense(len(self.list_of_classes))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.loss_computer = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        self.max_epochs_without_improvement = 6

    def call(self, x, training=False, logits=True):
        x = self.dense_layer_1(x)
        x = self.dropout_layer(x, training=training)
        x = self.dense_layer_2(x)

        if logits:
            return x
        else:
            return tf.nn.softmax(x)

    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = self.call(x_batch, training=True)
            loss_value = self.loss_computer(y_batch, y_pred)
        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

    def validation_step(self, validation_dataset):
        predictions = []
        labels = []
        for x_batch, y_batch in validation_dataset:
            y_pred = self.call(x_batch, training=False)
            labels.append(y_batch.numpy())
            predictions.append(y_pred.numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return labels, predictions

    def _compute_stats(self, one_hot_labels, predicted_logits):
        labels = self.list_of_classes[np.argmax(one_hot_labels, axis=1)]
        predictions = self.list_of_classes[np.argmax(predicted_logits, axis=1)]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro')
        return precision, recall, f1
        
    def train(self, training_dataset, validation_dataset):
        iteration = 0
        epoch = 0

        should_stop = False
        self.epochs_without_improvement = 0
        self.best_historic_loss = np.inf
        
        while not should_stop:
            for x_batch, y_batch in training_dataset:
                training_loss = self.train_step(x_batch, y_batch)
                iteration += 1
                if iteration % 1000 == 0:
                    print('iteration', iteration, 'training loss', training_loss)
            epoch += 1
            
            val_labels, val_predictions = self.validation_step(validation_dataset)
            val_precision, val_recall, val_f1 = self._compute_stats(val_labels, val_predictions)
            print(f'epoch {epoch} valstats f1 {val_f1:.3f} precision {val_precision:.3f} recall {val_recall:.3f}')
            
            should_stop = self._evaluate_training_stopper(-val_f1)  # minus because it expects a loss

    def _evaluate_training_stopper(self, current_validation_loss):
        if current_validation_loss < self.best_historic_loss:
            self.best_historic_loss = current_validation_loss
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.max_epochs_without_improvement:
            return True
        else:
            return False
