############################################################
# The FIRST PART IS TO CREATE THE DIRECTORIES AND MOVE FILES
############################################################
# This is to prepare the data
import os, shutil

# The path to the directry where the original dataset was uncompressed
original_dataset_dir = './original_full/train'

# The directory where we will store our smaller dataset
base_dir = './cats_and_dogs_small'
os.mkdir(base_dir)

# DIrectories for our training
# Validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# Directory with our dog training pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)

# Directory with our test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# Directory with our test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# Copy first 1000 cat images to train_cats_dir
fnames = {'cat.{}.jpg'.format(i) for i in range(1000)}
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = {'cat.{}.jpg'.format(i) for i in range(1000,1500)}
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

#Copy next 500 cat images to test_cats_dir
fnames = {'cat.{}.jpg'.format(i) for i in range(1500,2000)}
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src, dst)

# Copy first 1000 dogs images to train_dogs_dir
fnames = {'dog.{}.jpg'.format(i) for i in range(1000)}
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)


#Copy next 500 dog images to test_dog_dir
fnames = {'dog.{}.jpg'.format(i) for i in range(1000,1500)}
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

#Copy next 500 dog images to validation_dog_dir
fnames = {'dog.{}.jpg'.format(i) for i in range(1500,2000)}
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)
            
#Copy next 500 dog images to validation_dog_dir
fnames = {'dog.{}.jpg'.format(i) for i in range(1500,2000)}
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

################################################################
## The second part is to build the model and train it
################################################################
# This is the model
from keras import layers
from keras import models
from keras import optimizers

# To Generate data from images
from keras.preprocessing.image import ImageDataGenerator

# To draw
import matplotlib.pyplot as plt


# Prepare the model
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Prepare the data
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=29,
        class_mode='binary')


history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data = validation_generator,
        validation_steps=50)
model.save('cats_and_dogs_small_1.h5')

# Now, we're going to draw
acc     = history.history['acc']
val_acc = history.history['val_acc'] 
loss    = history.history['loss']
val_loss= history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc,'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Traning loss')
plt.plot(epochs, loss, 'b',  label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()



