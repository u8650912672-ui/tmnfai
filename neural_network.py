import tensorflow as tf
import numpy as np
import time
import os
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

class OsuNeuralNetwork:
    def __init__(self, input_shape=(128, 128, 4), learning_rate=0.001):
        """
        Neural network for TrackMania Nations Forever.
        :param input_shape: Input image dimensions (height, width, channels)
                            4 channels = last 4 frames stacked for temporal context
                            128x128 is sufficient for track-edge detection
        :param learning_rate: Learning rate
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()
        # deque auto-discards oldest entries — O(1) append/pop vs O(n) for list
        self.memory_buffer = deque(maxlen=10000)

    def _build_model(self):
        """
        Build the neural network architecture.
        :return: Compiled model
        """
        model = Sequential([
            # Convolutional layers for image processing
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),

            # Fully connected layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),

            # Output layer: [gas, steer_left, steer_right, brake]
            # Each output is a probability (0-1).
            # >0.5 = hold that key,  <=0.5 = release it.
            Dense(4, activation='sigmoid')
        ])

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        model.summary()
        return model

    def predict(self, state, epsilon=0.0):
        """
        Predict an action from the current screen state.
        :param state: Screen image (after preprocessing)
        :param epsilon: Exploration probability — adds Gaussian noise to encourage
                        the AI to try actions it hasn't seen before.
        :return: Predicted action [gas, steer_left, steer_right, brake]
        """
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)

        action = self.model.predict(state, verbose=0)[0]

        # Epsilon-greedy exploration: perturb action with Gaussian noise
        if epsilon > 0 and np.random.random() < epsilon:
            noise = np.random.normal(0, 0.2, action.shape)
            action = np.clip(action + noise, 0.0, 1.0)

        return action

    def add_to_memory(self, state, action, reward):
        """
        Add an experience to the replay buffer.
        :param state: State (screen image)
        :param action: Action [gas, steer_left, steer_right, brake]
        :param reward: Reward
        """
        # deque handles max size automatically
        self.memory_buffer.append((state, action, reward))

    def train(self, batch_size=32, epochs=1):
        """
        Train the model on stored experience.
        Uses advantage-weighted regression:
          - positive advantage  →  reinforce the action taken (target ≈ action)
          - negative advantage  →  pull action toward neutral 0.5 (unlearn it)
        :param batch_size: Batch size
        :param epochs: Number of epochs
        :return: Training history
        """
        if len(self.memory_buffer) < batch_size:
            print(f"Not enough data for training. Buffer size: {len(self.memory_buffer)}")
            return None

        indices = np.random.choice(len(self.memory_buffer), batch_size, replace=False)
        batch = [self.memory_buffer[i] for i in indices]

        states  = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch], dtype=np.float32)

        # Normalise rewards → per-batch advantages in [-1, 1]
        if rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / rewards.std()
        else:
            advantages = rewards - rewards.mean()
        advantages = np.clip(advantages, -1.0, 1.0)

        # Blend: high advantage → keep action, low advantage → pull toward 0.5
        alpha = ((advantages + 1.0) / 2.0).reshape(-1, 1)  # maps to [0, 1]
        neutral = np.full_like(actions, 0.5)
        targets = alpha * actions + (1.0 - alpha) * neutral

        history = self.model.fit(states, targets, epochs=epochs, verbose=0)
        return history

    def save_model(self, filepath="models/trackmania_model.h5"):
        """
        Save the model to a file.
        :param filepath: Save path
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="models/trackmania_model.h5"):
        """
        Load the model from a file.
        :param filepath: Path to the model file
        """
        if os.path.exists(filepath):
            self.model = load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found. Starting with a new model.")
