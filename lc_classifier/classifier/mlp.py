import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from typing import List


class MLP(Model):
    def __init__(self, list_of_classes: List[str]):
        super().__init__()
        self.list_of_classes = list_of_classes
        self.dense_layer_1 = Dense(
            1_000,
            activation='relu',
            kernel_initializer='he_uniform',
            name='dense_layer_1',
            # kernel_regularizer=tf.keras.regularizers.L1(l1=10.0)
        )
        self.dense_layer_2 = Dense(
            1_000,
            activation='relu',
            kernel_initializer='he_uniform',
            name='dense_layer_2',
            # kernel_regularizer=tf.keras.regularizers.L1(l1=10.0)
        )        
        self.dense_layer_3 = Dense(
            len(self.list_of_classes),
            name='dense_layer_3'
        )

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=5e-4,
            decay_steps=1_000,
            decay_rate=0.9,
            staircase=True)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_computer_true_labels = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.loss_computer_teacher_labels = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        self.max_epochs_without_improvement = 2

    def call(self, x, training=False, logits=True):
        x = self.dense_layer_1(x)
        x = self.dense_layer_2(x)
        x = self.dense_layer_3(x)

        if logits:
            return x
        else:
            return tf.nn.softmax(x)

    @tf.function
    def train_step(
        self, 
        x_batch: tf.Tensor, 
        y_batch: tf.Tensor, 
        y_teacher: tf.Tensor, 
        temperature: float, 
        teacher_weight: float
        ):
        with tf.GradientTape() as tape:
            # y_pred -> logits
            # y_batch -> one-hot labels
            # y_teacher -> probs
            y_pred = self.__call__(x_batch, training=True, logits=True)
            loss_value_true_labels = self.loss_computer_true_labels(y_batch, y_pred)
            loss_value_teacher_labels = self.loss_computer_teacher_labels(
                tf.nn.softmax(tf.math.log(y_teacher)/temperature, axis=1),
                tf.nn.softmax(y_pred/temperature, axis=1)
            )

            loss_value = (1.0 - teacher_weight) * loss_value_true_labels \
                + teacher_weight * loss_value_teacher_labels * temperature ** 2\
                + tf.keras.regularizers.L1(l1=1e-6)(self.dense_layer_1.kernel)\
                + tf.keras.regularizers.L1(l1=1e-6)(self.dense_layer_2.kernel)
            
        gradients = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss_value

    def validation_or_test_step(self, dataset: tf.data.Dataset) -> np.array:
        predictions = []
        labels = []
        for x_batch, y_batch in dataset:
            y_pred = self.__call__(x_batch, training=False)
            labels.append(y_batch.numpy())
            predictions.append(y_pred.numpy())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        return labels, predictions

    def _compute_stats(self, one_hot_labels: np.array, predicted_logits: np.array):
        labels = self.list_of_classes[np.argmax(one_hot_labels, axis=1)]
        predictions = self.list_of_classes[np.argmax(predicted_logits, axis=1)]
        print(classification_report(labels, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro')
        return precision, recall, f1
        
    def train(
            self, 
            training_dataset: tf.data.Dataset, 
            validation_dataset: tf.data.Dataset, 
            test_dataset: tf.data.Dataset):
        
        iteration = 0
        epoch = 0

        should_stop = False
        self.epochs_without_improvement = 0
        self.best_historic_loss = np.inf
        teacher_weight = 0.5
        for x_batch, y_batch, teacher_batch in training_dataset:
            training_loss = self.train_step(
                x_batch, y_batch, teacher_batch,
                temperature=1.0, teacher_weight=teacher_weight)
            iteration += 1
            if iteration % 100 == 0:
                teacher_weight = teacher_weight * 0.97
                print(
                    'iteration', iteration, 
                    'training loss', training_loss.numpy(),
                    f'teacher weight: {teacher_weight:.3f}',
                    f'lr {self.optimizer.learning_rate.numpy()}')

            if iteration % 2_500 == 0:
                epoch += 1
            
                val_labels, val_predictions = self.validation_or_test_step(validation_dataset)
                val_precision, val_recall, val_f1 = self._compute_stats(
                    val_labels, val_predictions)
                print(f'epoch {epoch} valstats f1 {val_f1:.3f} precision {val_precision:.3f} recall {val_recall:.3f}')
                
                test_labels, test_predictions = self.validation_or_test_step(test_dataset)
                test_precision, test_recall, test_f1 = self._compute_stats(
                    test_labels, test_predictions)
                print(f'epoch {epoch} valstats f1 {test_f1:.3f} precision {test_precision:.3f} recall {test_recall:.3f}')
                
                should_stop = self._evaluate_training_stopper(-val_f1)  # minus because it expects a loss
                if should_stop:
                    break

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
