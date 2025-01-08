import socket
import json
import numpy as np
import time
from collections import deque

import pydirectinput


class TrackmaniaEnvironment:
	def __init__(self):
		# OpenPlanet connection settings
		self.host = '127.0.0.1'
		self.port = 9000  # Adjust based on OpenPlanet plugin settings
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.connect_to_openplanet()

		# Track progression tracking
		self.last_position = None
		self.max_progress = 0  # Maximum progress reached on track
		self.wall_contacts = 0
		self.last_wall_contact_time = 0
		self.prev_positions = deque(maxlen=10)
		self.prev_speeds = deque(maxlen=5)

		# Reward weights
		self.reward_weights = {
			'speed': 1.0,  # Base speed reward
			'progress': 2.0,  # Track progression
			'wall_penalty': -5.0,  # Penalty for wall hits
			'checkpoint': 50.0,  # Bonus for checkpoint
			'direction': 1.0,  # Reward for going in right direction
			'finished': 1000.0  # Bonus for finishing
		}

	def connect_to_openplanet(self):
		"""Establish connection to OpenPlanet plugin"""
		try:
			self.socket.connect((self.host, self.port))
			print("Connected to OpenPlanet successfully")
		except Exception as e:
			print(f"Failed to connect to OpenPlanet: {e}")
			raise

	def get_telemetry(self):
		"""Get car telemetry data from OpenPlanet"""
		try:
			# Send request for data
			self.socket.send(b'GET_TELEMETRY')
			data = self.socket.recv(1024)
			telemetry = json.loads(data.decode())

			return {
				'position': np.array([
					telemetry['position']['x'],
					telemetry['position']['y'],
					telemetry['position']['z']
				]),
				'velocity': np.array([
					telemetry['velocity']['x'],
					telemetry['velocity']['y'],
					telemetry['velocity']['z']
				]),
				'speed': telemetry['speed'],
				'race_time': telemetry['race_time'],
				'checkpoint_count': telemetry['checkpoint_count'],
				'wall_contact': telemetry['wall_contact'],
				'finished': telemetry['finished']
			}
		except Exception as e:
			print(f"Failed to get telemetry: {e}")
			return None

	def calculate_track_progress(self, position):
		"""Calculate progress along the track"""
		if len(self.prev_positions) < 2:
			return 0.0

		# Calculate distance traveled along track
		total_distance = 0
		for i in range(len(self.prev_positions) - 1):
			total_distance += np.linalg.norm(
				self.prev_positions[i + 1] - self.prev_positions[i]
			)

		# Update maximum progress
		if total_distance > self.max_progress:
			self.max_progress = total_distance

		return total_distance

	def calculate_reward(self):
		"""Calculate reward based on car telemetry"""
		telemetry = self.get_telemetry()
		if telemetry is None:
			return 0.0

		reward = 0.0
		current_position = telemetry['position']
		current_speed = telemetry['speed']

		# 1. Speed reward
		speed_reward = self.reward_weights['speed'] * (current_speed / 100.0)  # Normalize by expected max speed
		reward += speed_reward

		# 2. Progress reward
		if self.last_position is not None:
			progress = self.calculate_track_progress(current_position)
			progress_reward = self.reward_weights['progress'] * (progress / 100.0)  # Normalize by track length
			reward += progress_reward

			# Direction reward - check if moving forward along track
			movement = current_position - self.last_position
			if len(self.prev_positions) >= 2:
				expected_direction = self.prev_positions[-1] - self.prev_positions[-2]
				direction_alignment = np.dot(movement, expected_direction)
				direction_reward = self.reward_weights['direction'] * max(0, direction_alignment)
				reward += direction_reward

		# 3. Wall contact penalty
		if telemetry['wall_contact']:
			current_time = time.time()
			# Only count wall hits that are at least 1 second apart
			if current_time - self.last_wall_contact_time > 1.0:
				self.wall_contacts += 1
				self.last_wall_contact_time = current_time
				reward += self.reward_weights['wall_penalty']

		# 4. Checkpoint bonus
		if telemetry['checkpoint_count'] > 0:
			reward += self.reward_weights['checkpoint']

		# 5. Finish bonus
		if telemetry['finished']:
			reward += self.reward_weights['finished']

		# Update history
		self.prev_positions.append(current_position)
		self.prev_speeds.append(current_speed)
		self.last_position = current_position

		return reward

	def step(self, action):
		"""Execute an action and return new state and reward"""
		# Execute action using your existing control method
		self.execute_action(action)

		# Get new state and calculate reward
		next_state = self.get_state()
		reward = self.calculate_reward()

		# Check if episode is done
		telemetry = self.get_telemetry()
		done = telemetry['finished'] if telemetry else False

		# Additional info for debugging
		info = {
			'speed': telemetry['speed'] if telemetry else 0,
			'wall_contacts': self.wall_contacts,
			'progress': self.max_progress
		}

		return next_state, reward, done, info

	def reset(self):
		"""Reset the environment"""
		# Your existing reset logic
		self.release_all_keys()
		pydirectinput.press(self.controls['reset'])
		pydirectinput.press(self.controls['reset'])

		# Reset tracking variables
		self.last_position = None
		self.max_progress = 0
		self.wall_contacts = 0
		self.last_wall_contact_time = 0
		self.prev_positions.clear()
		self.prev_speeds.clear()

		time.sleep(1.0)
		return self.get_state()