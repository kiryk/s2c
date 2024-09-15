#!/usr/bin/env python3

import sys
import math

import scipy.io.wavfile as wavfile
import scipy.signal     as signal
import numpy            as np
import cv2

from tqdm import tqdm

# Read the sound
print("Reading file")
name = sys.argv[1]

rate, data = wavfile.read(name)

maxrate = rate/2
focus = 1000

size = len(data)


chans = []
for _ in tqdm(data[0]):
    chans.append([])
for sample in tqdm(data):
    for i, v in enumerate(sample):
        chans[i].append(v)


# Create color filters
print("Creating filters")
freqs = [0, focus/3, focus/3, 2*focus/3, 2*focus/3, maxrate]
gains = [0, 0, 0, 0, 1, 1]
fred = signal.firwin2(201, freqs, gains, fs=rate)

gains = [0, 0, 1, 1, 0, 0]
fgreen = signal.firwin2(201, freqs, gains, fs=rate)

gains = [1, 1, 0, 0, 0, 0]
fblue = signal.firwin2(201, freqs, gains, fs=rate)


# Extract colors
print("Extracting colors")
red = signal.fftconvolve(fred, chans[0])
green = signal.fftconvolve(fgreen, chans[0])
blue = signal.fftconvolve(fblue, chans[0])


# Normalize colors
print("Normalizing colors")
print(red)
print(green)
print(blue)

red = abs(red)
green = abs(green)
blue = abs(blue)

prop = rate//30
full = size - size%prop
size = size//prop

red = np.mean(red[:full].reshape(-1, prop), axis=1)
green = np.mean(green[:full].reshape(-1, prop), axis=1)
blue = np.mean(blue[:full].reshape(-1, prop), axis=1)

for color in [red, green, blue]:
    mx = max(color)
    for i in tqdm(range(size)):
        color[i] = int((255 * (color[i] / mx)).item())


# Render
print("Rendering to", name+".avi")
def ctoimg(color):
    image = np.zeros((32, 32, 3), np.uint8)
    image[:,:] = color

    return image

frames = map(ctoimg, zip(red.astype(np.uint8), green.astype(np.uint8), blue.astype(np.uint8)))

video = cv2.VideoWriter(name+".avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    30,
    (32, 32))

for frame in tqdm(frames):
    video.write(frame)
