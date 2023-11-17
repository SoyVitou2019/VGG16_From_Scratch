import cv2
from keras.models import Sequential
from keras.layers import Conv2D
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./Cat.jpg')
img = cv2.resize(img, (224,224))



# cv2.imshow("frame", img)
# cv2.waitKey(0)


model = Sequential()
model.add(Conv2D(64, input_shape=(224,224,3), padding='same', kernel_size=(3,3)))
model.add(Conv2D(64,padding='same', kernel_size=(3,3)))

result = model.predict(np.array([img]))

model.build()
model.summary()


# Display features
for i in range(64):
    feature_map = result[0, :,:,  i]
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_map, cmap='gray')
    
plt.show()

