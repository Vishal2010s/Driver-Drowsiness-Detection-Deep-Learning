import os
import json
import hashlib
import shutil
import random
import sys
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from IPython.display import display
from tensorflow.keras import Model, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
print('TensorFlow:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))


## 2. Google Drive / dataset loadingif IN_COLAB:

drive.mount('/content/drive')
SOURCE_DIR = Path('/content/drive/MyDrive/project/archive/train') if IN_COLAB else Path('dataset')
OUTPUT_PARENT = Path('/content/drive/MyDrive/project/drowsiness_outputs') if IN_COLAB else Path('drowsiness_outputs')
if not SOURCE_DIR.is_dir():
    raise FileNotFoundError(f'Set SOURCE_DIR to the four-class source folder: {SOURCE_DIR}')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LAST_LAYERS = 30
FINE_TUNE_LR = 1e-5
ENABLE_FINE_TUNING = True
TASKS = {'eyes': ['Closed', 'Open'], 'yawn': ['no_yawn', 'yawn']}
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
RUN_DIR = OUTPUT_PARENT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=False)
SPLIT_ROOT = RUN_DIR / 'split_images'
SPLIT_ROOT.mkdir()
print('Output folder:', RUN_DIR)

# 3.Dataset Organizzation

# Validate source files before splitting. Keep the original per-class shuffle ratios.
classes = ['yawn', 'no_yawn', 'Open', 'Closed']
source_files = {}
hash_owner = {}
duplicates = []
for label in classes:
    folder = SOURCE_DIR / label
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing class folder: {folder}')
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.webp'})
    if len(images) < 10:
        raise ValueError(f'{label}: need at least 10 images for the 80/10/10 split')
    for path in images:
        with Image.open(path) as opened:
            opened.verify()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in hash_owner:
            duplicates.append({'file':str(path),'duplicate_of':hash_owner[digest]})
        else:
            hash_owner[digest] = str(path)
    source_files[label] = images
if duplicates:
    pd.DataFrame(duplicates).to_csv(RUN_DIR / 'duplicate_files.csv',index=False)
    raise ValueError('Exact duplicate files found. Review duplicate_files.csv and remove duplicates from the source before rerunning. Source files have not been modified.')


## 4. Train / Validation / Test split

# Creates split_images/train_split, split_images/validation and split_images/test, each with all four class folders. Source provenance is saved in split_manifest.csv.

random.seed(SEED)
manifest_rows = []
for label in classes:
    images = source_files[label].copy()
    random.shuffle(images)
    train_end = int(.8 * len(images))
    val_end = train_end + int(.1 * len(images))
    partitions = {'train_split':images[:train_end], 'validation':images[train_end:val_end], 'test':images[val_end:]}
    for split_name, paths in partitions.items():
        destination = SPLIT_ROOT / split_name / label
        destination.mkdir(parents=True)
        for path in paths:
            # Canonical RGB PNG keeps all source formats compatible with the
            # directory loader, and strips EXIF without rotating pixels.
            # Training and inference both use PIL RGB + nearest-neighbor resize.
            converted_name = path.name + '.png'
            with Image.open(path) as opened:
                opened.convert('RGB').save(destination / converted_name)
            manifest_rows.append({'source':str(path),'class':label,'split':split_name,'file':converted_name})
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(RUN_DIR / 'split_manifest.csv',index=False)
counts = pd.crosstab(manifest['class'],manifest['split'])
display(counts)
counts.to_csv(RUN_DIR / 'split_counts.csv')
assert len(manifest) == sum(len(paths) for paths in source_files.values())
TRAIN_DIR = str(SPLIT_ROOT / 'train_split')
VAL_DIR = str(SPLIT_ROOT / 'validation')
TEST_DIR = str(SPLIT_ROOT / 'test')

COPY_SPLITS_TO_LOCAL = False
if IN_COLAB and COPY_SPLITS_TO_LOCAL:
    local_split_root = Path('/content') / ('drowsiness_' + RUN_ID)
    shutil.copytree(SPLIT_ROOT, local_split_root)
    TRAIN_DIR = str(local_split_root / 'train_split')
    VAL_DIR = str(local_split_root / 'validation')
    TEST_DIR = str(local_split_root / 'test')


## 5. Image preprocessing
# Preprocessing is inline in generator creation and fixture validation.
# RGB, nearest-neighbor 224x224; CNN divides by 255; MobileNet uses preprocess_input.

## 6. Data augmentation
augmentation = {
    'cnn': ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=.10,
        brightness_range=(.9,1.1), horizontal_flip=True, fill_mode='nearest'),
    'mobilenet': ImageDataGenerator(preprocessing_function=preprocess_input,
        rotation_range=20, zoom_range=.20, brightness_range=(.8,1.2), horizontal_flip=True, fill_mode='nearest')}
evaluation_preprocessing = {
    'cnn': ImageDataGenerator(rescale=1./255),
    'mobilenet': ImageDataGenerator(preprocessing_function=preprocess_input)}


## 7. CNN training

training_records = {}
for task in TASKS:
    model_type = 'cnn'
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    name = task + '_' + model_type
    model_folder = RUN_DIR / name
    model_folder.mkdir(exist_ok=True)
    train_generator = augmentation[model_type].flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, classes=TASKS[task],
        class_mode='categorical', shuffle=True, seed=SEED, interpolation='nearest')
    validation_generator = evaluation_preprocessing[model_type].flow_from_directory(
        VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, classes=TASKS[task],
        class_mode='categorical', shuffle=False, interpolation='nearest')
    test_generator = evaluation_preprocessing[model_type].flow_from_directory(
        TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, classes=TASKS[task],
        class_mode='categorical', shuffle=False, interpolation='nearest')
    assert train_generator.class_indices == validation_generator.class_indices == test_generator.class_indices
    task_classes = TASKS[task]
    images, labels = next(train_generator)
    print(name, 'image shape:', images.shape, 'label shape:', labels.shape,
          'pixel range:', float(images.min()), float(images.max()))
    preprocessing_record = pd.DataFrame([{
        'model': name, 'image_height': IMAGE_SIZE[0], 'image_width': IMAGE_SIZE[1],
        'batch_size': BATCH_SIZE, 'pixel_min': float(images.min()), 'pixel_max': float(images.max()),
        'train_images': train_generator.samples, 'validation_images': validation_generator.samples,
        'test_images': test_generator.samples, 'class_indices': str(train_generator.class_indices)}])
    preprocessing_record.to_csv(model_folder / 'preprocessing_summary.csv', index=False)

    fig, axes = plt.subplots(2,4,figsize=(12,6))
    for i, axis in enumerate(axes.flat):
        axis.axis('off')
        if i < len(images):
            shown = (images[i]+1)/2 if model_type == 'mobilenet' else images[i]
            axis.imshow(np.clip(shown,0,1))
            axis.set_title(task_classes[int(labels[i].argmax())])
    fig.tight_layout()
    fig.savefig(model_folder / 'augmented_samples.png')
    plt.show()
    plt.close(fig)
    train_generator.reset()
    if model_type == 'cnn':
        model = Sequential([Input(shape=(224, 224, 3)), Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), GlobalAveragePooling2D(), Dense(256, activation='relu'), Dropout(0.3), Dense(128, activation='relu'), Dropout(0.2), Dense(2, activation='softmax')])
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        inputs = Input(shape=(224, 224, 3))
        # Keep backbone BatchNormalization statistics in inference mode.
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
    import io
    from contextlib import redirect_stdout
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        model.summary()
    summary_text = summary_buffer.getvalue()
    print(summary_text)
    (model_folder / 'model_summary.txt').write_text(summary_text)

    model.compile(optimizer=Adam(learning_rate=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
    checkpoint = RUN_DIR / ('best_' + name + '.keras')
    early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=.2,patience=3,verbose=1)
    checkpoint_callback = ModelCheckpoint(filepath=str(checkpoint),monitor='val_loss',save_best_only=True,verbose=1)
    history = model.fit(train_generator,validation_data=validation_generator,epochs=EPOCHS,
                        callbacks=[early_stop,reduce_lr,checkpoint_callback])
    fine_tuning_selected = False
    if model_type == 'mobilenet' and ENABLE_FINE_TUNING:
        # Start from the selected frozen-stage checkpoint, never final weights.
        model = load_model(checkpoint, compile=False)
        backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
        backbone.trainable = True
        for layer in backbone.layers[:-FINE_TUNE_LAST_LAYERS]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        fine_checkpoint = RUN_DIR / ('fine_' + name + '.keras')
        fine_history = model.fit(
            train_generator, validation_data=validation_generator,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=3),
                       ModelCheckpoint(str(fine_checkpoint), monitor='val_loss', save_best_only=True)])
        pd.DataFrame(fine_history.history).to_csv(model_folder / 'fine_tuning_history.csv', index=False)
        frozen_model = load_model(checkpoint, compile=False)
        fine_model = load_model(fine_checkpoint, compile=False)
        for candidate in (frozen_model, fine_model):
            candidate.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        validation_generator.reset()
        frozen_loss = float(frozen_model.evaluate(validation_generator, verbose=0)[0])
        validation_generator.reset()
        fine_loss = float(fine_model.evaluate(validation_generator, verbose=0)[0])
        fine_tuning_selected = fine_loss < frozen_loss
        pd.DataFrame([{'stage':'frozen','val_loss':frozen_loss},
                      {'stage':'fine_tuned','val_loss':fine_loss}]).to_csv(
                          model_folder / 'stage_selection.csv', index=False)
        if fine_tuning_selected:
            shutil.copyfile(fine_checkpoint, checkpoint)
        print(name, 'fine-tuned stage selected:', fine_tuning_selected)

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(model_folder / 'training_history.csv',index=False)
    best_epoch = int(np.argmin(history.history['val_loss']))
    print(name,'best checkpoint epoch:',best_epoch+1,'validation loss:',history.history['val_loss'][best_epoch])
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    history_frame[['accuracy','val_accuracy']].plot(ax=axes[0],title=name+' accuracy')
    history_frame[['loss','val_loss']].plot(ax=axes[1],title=name+' loss')
    fig.tight_layout()
    fig.savefig(model_folder / 'training_curves.png')
    plt.show()
    plt.close(fig)


    training_records[name] = {'task':task,'model_type':model_type,'checkpoint':checkpoint,'folder':model_folder,'fine_tuning_selected':fine_tuning_selected}


## 8. MobileNetV2 training 

for task in TASKS:
    model_type = 'mobilenet'
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    name = task + '_' + model_type
    model_folder = RUN_DIR / name
    model_folder.mkdir(exist_ok=True)
    train_generator = augmentation[model_type].flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, classes=TASKS[task],
        class_mode='categorical', shuffle=True, seed=SEED, interpolation='nearest')
    validation_generator = evaluation_preprocessing[model_type].flow_from_directory(
        VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, classes=TASKS[task],
        class_mode='categorical', shuffle=False, interpolation='nearest')
    test_generator = evaluation_preprocessing[model_type].flow_from_directory(
        TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, classes=TASKS[task],
        class_mode='categorical', shuffle=False, interpolation='nearest')
    assert train_generator.class_indices == validation_generator.class_indices == test_generator.class_indices
    task_classes = TASKS[task]
    images, labels = next(train_generator)
    print(name, 'image shape:', images.shape, 'label shape:', labels.shape,
          'pixel range:', float(images.min()), float(images.max()))
    preprocessing_record = pd.DataFrame([{
        'model': name, 'image_height': IMAGE_SIZE[0], 'image_width': IMAGE_SIZE[1],
        'batch_size': BATCH_SIZE, 'pixel_min': float(images.min()), 'pixel_max': float(images.max()),
        'train_images': train_generator.samples, 'validation_images': validation_generator.samples,
        'test_images': test_generator.samples, 'class_indices': str(train_generator.class_indices)}])
    preprocessing_record.to_csv(model_folder / 'preprocessing_summary.csv', index=False)

    fig, axes = plt.subplots(2,4,figsize=(12,6))
    for i, axis in enumerate(axes.flat):
        axis.axis('off')
        if i < len(images):
            shown = (images[i]+1)/2 if model_type == 'mobilenet' else images[i]
            axis.imshow(np.clip(shown,0,1))
            axis.set_title(task_classes[int(labels[i].argmax())])
    fig.tight_layout()
    fig.savefig(model_folder / 'augmented_samples.png')
    plt.show()
    plt.close(fig)
    train_generator.reset()
    if model_type == 'cnn':
        model = Sequential([Input(shape=(224, 224, 3)), Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'), BatchNormalization(), MaxPooling2D(pool_size=(2, 2)), GlobalAveragePooling2D(), Dense(256, activation='relu'), Dropout(0.3), Dense(128, activation='relu'), Dropout(0.2), Dense(2, activation='softmax')])
    else:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_model.trainable = False
        inputs = Input(shape=(224, 224, 3))
        # Keep backbone BatchNormalization statistics in inference mode.
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
    import io
    from contextlib import redirect_stdout
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        model.summary()
    summary_text = summary_buffer.getvalue()
    print(summary_text)
    (model_folder / 'model_summary.txt').write_text(summary_text)

    model.compile(optimizer=Adam(learning_rate=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])
    checkpoint = RUN_DIR / ('best_' + name + '.keras')
    early_stop = EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=.2,patience=3,verbose=1)
    checkpoint_callback = ModelCheckpoint(filepath=str(checkpoint),monitor='val_loss',save_best_only=True,verbose=1)
    history = model.fit(train_generator,validation_data=validation_generator,epochs=EPOCHS,
                        callbacks=[early_stop,reduce_lr,checkpoint_callback])
    fine_tuning_selected = False
    if model_type == 'mobilenet' and ENABLE_FINE_TUNING:
        # Start from the selected frozen-stage checkpoint, never final weights.
        model = load_model(checkpoint, compile=False)
        backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model))
        backbone.trainable = True
        for layer in backbone.layers[:-FINE_TUNE_LAST_LAYERS]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=Adam(learning_rate=FINE_TUNE_LR),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        fine_checkpoint = RUN_DIR / ('fine_' + name + '.keras')
        fine_history = model.fit(
            train_generator, validation_data=validation_generator,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=3),
                       ModelCheckpoint(str(fine_checkpoint), monitor='val_loss', save_best_only=True)])
        pd.DataFrame(fine_history.history).to_csv(model_folder / 'fine_tuning_history.csv', index=False)
        frozen_model = load_model(checkpoint, compile=False)
        fine_model = load_model(fine_checkpoint, compile=False)
        for candidate in (frozen_model, fine_model):
            candidate.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        validation_generator.reset()
        frozen_loss = float(frozen_model.evaluate(validation_generator, verbose=0)[0])
        validation_generator.reset()
        fine_loss = float(fine_model.evaluate(validation_generator, verbose=0)[0])
        fine_tuning_selected = fine_loss < frozen_loss
        pd.DataFrame([{'stage':'frozen','val_loss':frozen_loss},
                      {'stage':'fine_tuned','val_loss':fine_loss}]).to_csv(
                          model_folder / 'stage_selection.csv', index=False)
        if fine_tuning_selected:
            shutil.copyfile(fine_checkpoint, checkpoint)
        print(name, 'fine-tuned stage selected:', fine_tuning_selected)

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(model_folder / 'training_history.csv',index=False)
    best_epoch = int(np.argmin(history.history['val_loss']))
    print(name,'best checkpoint epoch:',best_epoch+1,'validation loss:',history.history['val_loss'][best_epoch])
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    history_frame[['accuracy','val_accuracy']].plot(ax=axes[0],title=name+' accuracy')
    history_frame[['loss','val_loss']].plot(ax=axes[1],title=name+' loss')
    fig.tight_layout()
    fig.savefig(model_folder / 'training_curves.png')
    plt.show()
    plt.close(fig)


    training_records[name] = {'task':task,'model_type':model_type,'checkpoint':checkpoint,'folder':model_folder,'fine_tuning_selected':fine_tuning_selected}


## 9. Final evaluation

# Evaluate the validation-selected checkpoint once on the held-out test set. Review per-class recall and the confusion matrix, not accuracy alone.

evaluations = {}
all_metrics = []
for name, record in training_records.items():
    task, model_type = record['task'], record['model_type']
    model = load_model(record['checkpoint'])
    test_generator = evaluation_preprocessing[model_type].flow_from_directory(
        TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, classes=TASKS[task],
        class_mode='categorical', shuffle=False, interpolation='nearest')
    loss, accuracy = model.evaluate(test_generator, verbose=1)
    test_generator.reset()
    probabilities = model.predict(test_generator, verbose=1)
    actual = test_generator.classes
    predicted = probabilities.argmax(axis=1)
    report = classification_report(actual, predicted, labels=[0,1], target_names=TASKS[task], output_dict=True, zero_division=0)
    table = pd.DataFrame({'file':test_generator.filenames,'actual':[TASKS[task][i] for i in actual],
                          'predicted':[TASKS[task][i] for i in predicted]})
    for i, label in enumerate(TASKS[task]):
        table['p_'+label] = probabilities[:,i]
    table.to_csv(record['folder'] / 'test_predictions.csv', index=False)
    all_metrics.append({'task':task,'model':model_type,'test_accuracy':float(accuracy),
                        'test_loss':float(loss),'macro_f1':report['macro avg']['f1-score']})
    evaluations[name] = {'actual':actual,'predicted':predicted,'report':report,
                         'fixture':test_generator.filepaths[0],'probabilities':probabilities[0].tolist()}


## 10. Confusion matrix / classification report

for name, result in evaluations.items():
    record = training_records[name]
    labels = TASKS[record['task']]
    matrix = confusion_matrix(result['actual'], result['predicted'], labels=[0,1])
    pd.DataFrame(matrix,index=labels,columns=labels).to_csv(record['folder'] / 'confusion_matrix.csv')
    report = pd.DataFrame(result['report']).transpose()
    report.to_csv(record['folder'] / 'classification_report.csv')
    display(report)
    fig, axis = plt.subplots(figsize=(5,4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set(xlabel='Predicted', ylabel='Actual', title=name)
    fig.tight_layout()
    fig.savefig(record['folder'] / 'confusion_matrix.png')
    plt.show()
    plt.close(fig)


## 11. Compare CNN vs MobileNet

# Test metrics describe the final models. They are not used to select or tune a deployment model; the dashboard allows an explicit architecture choice.
comparison = pd.DataFrame(all_metrics)
comparison.to_csv(RUN_DIR / 'model_comparison.csv', index=False)
display(comparison)
fig, axes = plt.subplots(1,2,figsize=(11,4))
for axis, metric in zip(axes, ['test_accuracy','macro_f1']):
    comparison.pivot(index='task',columns='model',values=metric).plot.bar(ax=axis, ylim=(0,1), title=metric)
fig.tight_layout()
fig.savefig(RUN_DIR / 'model_comparison.png')
plt.show()
plt.close(fig)

## 12. Save best models

metadata = {'version':1,'tensorflow':tf.__version__,'image_size':[224,224],
            'resize':'nearest','color':'RGB','orientation':'stored_pixels_no_exif_rotation','models':{}}
for task in TASKS:
    metadata['models'][task] = {}
    for model_type in ['cnn','mobilenet']:
        name = task+'_'+model_type
        record, result = training_records[name], evaluations[name]
        fixture = RUN_DIR / (name+'_example.png')
        shutil.copyfile(result['fixture'], fixture)
        model = load_model(record['checkpoint'], compile=False)
        with Image.open(fixture) as opened:
            fixture_array = np.asarray(opened.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.NEAREST), dtype=np.float32)
        fixture_array = fixture_array / 255.0 if model_type == 'cnn' else preprocess_input(fixture_array)
        expected = model.predict(fixture_array[None], verbose=0)[0]
        np.testing.assert_allclose(expected,result['probabilities'],rtol=1e-4,atol=1e-5)
        metadata['models'][task][model_type] = {
            'file':record['checkpoint'].name,'sha256':hashlib.sha256(record['checkpoint'].read_bytes()).hexdigest(),
            'classes':TASKS[task],'preprocessing':'divide_255' if model_type=='cnn' else 'mobilenet_v2',
            'example':fixture.name,'expected_probabilities':expected.tolist()}
(RUN_DIR / 'metadata.json').write_text(json.dumps(metadata,indent=2))
(RUN_DIR / 'run_config.json').write_text(json.dumps({'source_dir':str(SOURCE_DIR),'seed':SEED,
    'epochs':EPOCHS,'batch_size':BATCH_SIZE,'image_size':list(IMAGE_SIZE),'split_ratios':[.8,.1,.1],
    'subject_disjoint':False,'mobilenet_fine_tuning_enabled':ENABLE_FINE_TUNING,
    'fine_tune_epochs':FINE_TUNE_EPOCHS,'fine_tune_lr':FINE_TUNE_LR,
    'selected_stage':{name:('fine_tuned' if record['fine_tuning_selected'] else 'baseline') for name,record in training_records.items()}},indent=2))
(RUN_DIR / 'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('Saved all four best checkpoints, metadata and inference parity examples:', RUN_DIR)


## 13. Eye + Yawn prediction logic

# A manual crop of one eye is required. The original portrait goes to the yawn model. The 0.80 confidence cutoff is an uncalibrated demonstration setting.

# Optional single portrait prediction, kept separate from reporting.
PORTRAIT_PATH = None  # set to your portrait path after training
EYE_BOX = None  # (left, top, right, bottom), one eye including eyelids
if PORTRAIT_PATH is not None:
    with Image.open(PORTRAIT_PATH) as opened:
        portrait = opened.convert('RGB')
    if EYE_BOX is None:
        raise ValueError('Set EYE_BOX for the optional notebook eye demonstration.')
    left, top, right, bottom = EYE_BOX
    if not (0 <= left < right <= portrait.width and 0 <= top < bottom <= portrait.height):
        raise ValueError('Invalid eye coordinates')
    observation = {}
    for task, image in [('eyes', portrait.crop(EYE_BOX)), ('yawn', portrait)]:
        spec = metadata['models'][task]['mobilenet']
        model = load_model(RUN_DIR / spec['file'], compile=False)
        array = np.asarray(image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.NEAREST), dtype=np.float32)
        probabilities = model(preprocess_input(array)[None], training=False).numpy()[0]
        index = int(probabilities.argmax())
        observation[task] = spec['classes'][index] if probabilities[index] >= .80 else 'Uncertain'
    display(pd.DataFrame([observation]))


## 16: Decision Fusion Logic

# Map Open/no_yawn → 0 Alert, yawn → 1 Mild Fatigue, Closed → 2 Severe Fatigue for each held-out sample. 

fatigue_mapping = {'Open': 0, 'no_yawn': 0, 'yawn': 1, 'Closed': 2}
fatigue_labels = {0: 'Alert', 1: 'Mild Fatigue', 2: 'Severe Fatigue'}
fusion_tables = {}
for architecture in ['cnn', 'mobilenet']:
    tables = []
    for task in TASKS:
        frame = pd.read_csv(RUN_DIR / f'{task}_{architecture}' / 'test_predictions.csv')
        frame['task'] = task
        tables.append(frame)
    results = pd.concat(tables, ignore_index=True)
    results['reference_stage'] = results['actual'].map(fatigue_mapping)
    results['fatigue_stage'] = results['predicted'].map(fatigue_mapping)
    results['fatigue_label'] = results['fatigue_stage'].map(fatigue_labels)
    summary = results.groupby(['predicted', 'fatigue_stage', 'fatigue_label']).size().reset_index(name='count')
    print(architecture, 'decision fusion summary')
    display(summary)
    results.to_csv(RUN_DIR / f'decision_fusion_{architecture}.csv', index=False)
    summary.to_csv(RUN_DIR / f'decision_summary_{architecture}.csv', index=False)
    fusion_tables[architecture] = results

if PORTRAIT_PATH is not None:
    if observation['eyes'] == 'Closed':
        fatigue_stage = 2
    elif observation['eyes'] == 'Open' and observation['yawn'] == 'yawn':
        fatigue_stage = 1
    elif observation['eyes'] == 'Open' and observation['yawn'] == 'no_yawn':
        fatigue_stage = 0
    else:
        fatigue_stage = None
    print('Paired observation stage:', fatigue_labels.get(fatigue_stage, 'Uncertain'))


## 17: Driver Fatigue Progression Curve
FRAMES_PER_INTERVAL = 10
progression_tables = {}
for architecture, results in fusion_tables.items():
    model_label = 'Custom CNN' if architecture == 'cnn' else 'MobileNetV2'
    alert = results.loc[results['fatigue_stage'] == 0].copy()
    mild = results.loc[results['fatigue_stage'] == 1].copy()
    severe = results.loc[results['fatigue_stage'] == 2].copy()
    early_alert = int(len(alert) * .60)
    early_mild = int(len(mild) * .40)
    early_severe = int(len(severe) * .30)
    simulated = pd.concat([
        alert.iloc[:early_alert], mild.iloc[:early_mild],
        alert.iloc[early_alert:], mild.iloc[early_mild:],
        severe.iloc[:early_severe], severe.iloc[early_severe:]
    ], ignore_index=True)
    if simulated.empty:
        print(model_label, ': no predictions available')
        continue
    assert len(simulated) == len(results)
    simulated['interval'] = np.arange(len(simulated)) // FRAMES_PER_INTERVAL
    summary = simulated.groupby('interval').agg(
        fatigue_score=('fatigue_stage', 'mean'), frame_count=('file', 'size')).reset_index()
    summary['stage'] = np.select(
        [summary['fatigue_score'] < .5, summary['fatigue_score'] < 1.5],
        ['Alert', 'Mild Fatigue'], default='Severe Fatigue')
    summary['simulation_type'] = 'constructed from predicted stages'
    print(model_label, '- constructed simulation, ten observations per interval')
    display(summary)
    transitions = summary.loc[summary['stage'] != summary['stage'].shift()].copy().iloc[1:]
    transitions['from_stage'] = summary['stage'].shift().loc[transitions.index]
    display(transitions[['interval', 'from_stage', 'stage']])
    for before, after in [('Alert', 'Mild Fatigue'), ('Mild Fatigue', 'Severe Fatigue')]:
        matching = transitions.loc[(transitions['from_stage'] == before) & (transitions['stage'] == after)]
        print(before, '→', after, ':', 'not observed' if matching.empty else f"interval {int(matching.iloc[0]['interval'])}")
    simulated.to_csv(RUN_DIR / f'fatigue_session_{architecture}.csv', index=False)
    summary.to_csv(RUN_DIR / f'fatigue_intervals_{architecture}.csv', index=False)
    transitions.to_csv(RUN_DIR / f'fatigue_transitions_{architecture}.csv', index=False)
    fig, axis = plt.subplots(figsize=(12,6))
    axis.plot(summary['interval'], summary['fatigue_score'], marker='o', linewidth=3, color='#1f77b4')
    axis.set(title=f'{model_label} Driver Fatigue Progression Curve',
             xlabel='Driving Time Interval', ylabel='Fatigue Level',
             yticks=[0,1,2], yticklabels=['Alert','Mild Fatigue','Severe Fatigue'], ylim=(-.1,2.1))
    axis.grid(True, alpha=.65)
    fig.text(.5, .01, 'Constructed simulation: 10 observations per interval; not a recorded driving session.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(RUN_DIR / f'fatigue_progression_{architecture}.png', dpi=160, bbox_inches='tight')
    plt.show(); plt.close(fig)
    progression_tables[architecture] = summary

## 18: Performance Analysis

class_rows = []
error_tables = []
for name, result in evaluations.items():
    record = training_records[name]
    for label in TASKS[record['task']]:
        values = result['report'][label]
        class_rows.append({'task': record['task'], 'model': record['model_type'], 'class': label,
                           **{metric: values[metric] for metric in ['precision','recall','f1-score','support']}})
    table = pd.read_csv(record['folder'] / 'test_predictions.csv')
    errors = table.loc[table['actual'] != table['predicted']].copy()
    errors['model'] = name
    error_tables.append(errors)
performance = pd.DataFrame(class_rows)
display(performance)
performance.to_csv(RUN_DIR / 'performance_by_class.csv', index=False)
errors = pd.concat(error_tables, ignore_index=True)
display(errors.groupby('model').head(5))
errors.to_csv(RUN_DIR / 'performance_error_cases.csv', index=False)
for metric in ['precision','recall','f1-score']:
    fig, axis = plt.subplots(figsize=(9,4))
    performance.pivot(index='class', columns='model', values=metric).plot.bar(ax=axis)
    axis.set(title=f'Per-class {metric}', ylabel=metric, ylim=(0,1))
    axis.tick_params(axis='x', rotation=0); fig.tight_layout()
    fig.savefig(RUN_DIR / f'performance_{metric}.png', dpi=160)
    plt.show(); plt.close(fig)

preprocessing_tables = [pd.read_csv(record['folder'] / 'preprocessing_summary.csv') for record in training_records.values()]
pd.concat(preprocessing_tables, ignore_index=True).to_csv(RUN_DIR / 'preprocessing_summary.csv', index=False)


accuracy_rows = []
for architecture in ['cnn', 'mobilenet']:
    correct = 0
    total = 0
    for task in TASKS:
        predictions = pd.read_csv(RUN_DIR / f'{task}_{architecture}' / 'test_predictions.csv')
        correct += int((predictions['actual'] == predictions['predicted']).sum())
        total += len(predictions)
    accuracy_rows.append({'model': 'Custom CNN' if architecture == 'cnn' else 'MobileNetV2',
                          'correct': correct, 'test_images': total,
                          'test_accuracy_percent': 100 * correct / total if total else np.nan})
accuracy_comparison = pd.DataFrame(accuracy_rows)
display(accuracy_comparison)
accuracy_comparison.to_csv(RUN_DIR / 'model_accuracy_comparison.csv', index=False)
fig, axis = plt.subplots(figsize=(7,6))
bars = axis.bar(accuracy_comparison['model'], accuracy_comparison['test_accuracy_percent'], color='#1f77b4')
axis.set(title='Model Accuracy Comparison', ylabel='Test Accuracy (%)', ylim=(0,100))
for bar in bars:
    value = bar.get_height()
    if np.isfinite(value):
        axis.annotate(f'{value:.2f}%', (bar.get_x() + bar.get_width()/2, value),
                      xytext=(0,4), textcoords='offset points', ha='center', fontweight='bold')
fig.text(.5,.01,'Sample-weighted summary of separate eye and yawn test sets.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,1))
fig.savefig(RUN_DIR / 'model_accuracy_comparison.png', dpi=160, bbox_inches='tight')
plt.show(); plt.close(fig)


## 19. Streamlit deployment

# The dashboard is a SEPARATE Python file; run it locally after downloading.
# pip install -r requirements_dashboard.txt
# streamlit run driver_drowsiness_dashboard.py
# Select the extracted RUN_DIR folder in the dashboard sidebar.
# Lightweight bundle includes checkpoints, reports, metadata and sample images.
# Split images remain in RUN_DIR/split_images on Drive to avoid a huge download.
archive = RUN_DIR / 'dashboard_bundle.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
    for path in RUN_DIR.rglob('*'):
        if path.is_file() and 'split_images' not in path.relative_to(RUN_DIR).parts and path != archive:
            zipped.write(path, arcname=str(path.relative_to(RUN_DIR)))
print('Dashboard bundle:',archive)
print('Split image folders:',SPLIT_ROOT)
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))


