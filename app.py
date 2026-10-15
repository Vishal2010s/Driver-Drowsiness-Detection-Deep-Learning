import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

from PIL import Image

import streamlit as st

import tensorflow as tf

import cv2

st.title('Driver eye and yawn dashboard')

root = Path(st.sidebar.text_input('Extracted trained bundle folder',

    str(Path(__file__).resolve().parent / 'trained_bundle'), key='bundle_folder')).expanduser()

kind = 'mobilenet'

threshold = .70

reports, prediction = st.tabs(['Training reports', 'Prediction'])

with reports:

    st.subheader('Dataset and preprocessing')
    st.caption('preprocessing_summary.csv')
    preprocessing_path = root / 'preprocessing_summary.csv'
    if preprocessing_path.exists():
        try:
            st.dataframe(pd.read_csv(preprocessing_path), hide_index=True, use_container_width=True)
        except pd.errors.EmptyDataError:
            st.info('The preprocessing summary has no saved rows.')
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            st.warning(f'Cannot read preprocessing_summary.csv: {exc}')
    else:
        st.info('Add preprocessing_summary.csv from your saved bundle to display dataset and preprocessing details.')

    st.subheader('CNN and MobileNetV2 comparison')

    # path = root / 'model_comparison.csv'

    # if path.exists():

    #     comparison = pd.read_csv(path)

    #     simple = comparison[['task', 'model', 'test_accuracy']].copy()

    #     simple['test_accuracy'] = (simple['test_accuracy'] * 100).round(2)

    #     simple.columns = ['Task', 'Model', 'Test accuracy (%)']

    #     st.dataframe(simple, hide_index=True, use_container_width=True)

    #     st.bar_chart(comparison.pivot(index='task', columns='model', values='test_accuracy'))

    # else:

    #     st.info('Add model_comparison.csv from your saved bundle.')

    task = st.selectbox('View results for', ['eyes', 'yawn'], key='report_task')

    columns = st.columns(2)

    for column, architecture in zip(columns, ['cnn', 'mobilenet']):

        with column:

            st.markdown('**Custom CNN**' if architecture == 'cnn' else '**MobileNetV2**')

            folder = root / f'{task}_{architecture}'

            for filename, caption in [('training_curves.png', 'Training accuracy and loss'),

                                      ('confusion_matrix.png', 'Confusion matrix')]:

                path = folder / filename

                if path.exists():

                    st.image(str(path), caption=caption)

            with st.expander('Precision and recall and F1-score'):

                path = folder / 'classification_report.csv'

                if path.exists():

                    table = pd.read_csv(path, index_col=0)

                    classes = ['Closed', 'Open'] if task == 'eyes' else ['no_yawn', 'yawn']

                    st.dataframe(table.loc[classes, ['precision', 'recall', 'f1-score']].round(3),use_container_width=True)

    st.subheader('Decision fusion and simulated fatigue progression')
    
    st.caption('Predictions are arranged by stage for this constructed simulation. Each interval contains ten observations; this is not a recorded driving session.')

    fatigue_columns = st.columns(2)

    for column, architecture in zip(fatigue_columns, ['cnn', 'mobilenet']):

        with column:

            st.markdown(f'**{architecture.upper()}**')

            for filename in [f'decision_summary_{architecture}.csv']:

                path = root / filename

                if path.exists():

                    try:

                        st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)

                    except pd.errors.EmptyDataError:

                        st.caption('No recorded transitions.')

            plot = root / f'fatigue_progression_{architecture}.png'

            if plot.exists():

                st.image(str(plot))

            else:

                st.caption('Run the combined notebook to generate this progression plot.')

    st.subheader('Performance analysis')
    
    accuracy_plot = root / 'model_accuracy_comparison.png'

    accuracy_table = root / 'model_accuracy_comparison.csv'

    if accuracy_table.exists():
    
        st.dataframe(pd.read_csv(accuracy_table), hide_index=True, use_container_width=True)

    if accuracy_plot.exists():

        st.image(str(accuracy_plot))

        st.caption('Sample-weighted accuracy across separate eye and yawn test sets; not fused fatigue accuracy.')


    for filename in ['robustness_accuracy.png']:

        path = root / filename

        if path.exists():

            st.image(str(path))
    for filename in ['robustness_results.csv']:
    
            path = root / filename
    
            if path.exists():
    
                st.dataframe(pd.read_csv(path), hide_index=True, use_container_width=True)

    # errors_path = root / 'performance_error_cases.csv'

    # if errors_path.exists():

    #     with st.expander('Misclassified test images'):

    #         st.dataframe(pd.read_csv(errors_path), hide_index=True, use_container_width=True)

    st.caption('Robustness plots appear only after the optional experiment is run. Synthetic transformations do not establish accuracy across people or camera conditions.')

    with st.expander('Limitations and error cases'):

        st.write('Results can change with lighting, pose, glasses, camera framing and automatic eye selection. A single image cannot establish sustained fatigue.')

        st.write('The existing image split does not establish performance on independent people or sessions. Reported accuracy is specific to the saved test set.')

        st.write('Measured robustness requires labelled tests under different conditions; it is not established by these plots alone.')

with prediction:

    try:

        metadata_text = (root / 'metadata.json').read_text(encoding='utf-8')

        metadata = json.loads(metadata_text)

        fingerprints = {}

        for task in ['eyes', 'yawn']:

            spec = metadata['models'][task]['mobilenet']

            fingerprints[task + '_mobilenet'] = hashlib.sha256((root / spec['file']).read_bytes()).hexdigest()

        signature = (str(root.resolve()), metadata_text, tuple(sorted(fingerprints.items())))

        # Per-session reuse without custom cache functions; reload if metadata/weights change.

        if st.session_state.get('model_signature') != signature:

            if metadata.get('version') != 1 or metadata.get('resize') != 'nearest' or metadata.get('image_size') != [224, 224]:

                raise ValueError('Unsupported bundle preprocessing contract')

            models = {}

            expected_tasks = [('eyes', ['Closed', 'Open']), ('yawn', ['no_yawn', 'yawn'])]

            for task, expected_classes in expected_tasks:

                for kind in ['mobilenet']:

                    spec = metadata['models'][task][kind]

                    if spec['classes'] != expected_classes:

                        raise ValueError(f"{task}_{kind}: unexpected class order {spec['classes']} != {expected_classes}")

                    key = task + '_' + kind

                    if fingerprints[key] != spec['sha256']:

                        raise ValueError(f'{key}: model hash differs from metadata')

                    model_path = root / spec['file']

                    if not model_path.exists():

                        raise FileNotFoundError(f'Missing model file: {model_path}')

                    model = tf.keras.models.load_model(model_path, compile=False)

                    if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != 2:

                        raise ValueError(f'{key}: unexpected model shape')

                    example_path = root / spec['example']

                    if not example_path.exists():

                        raise FileNotFoundError(f'Missing model validation example: {example_path}')

                    with Image.open(example_path) as fixture:

                        p = model(tf.keras.applications.mobilenet_v2.preprocess_input(np.asarray(fixture.convert('RGB').resize((224,224), Image.Resampling.NEAREST), dtype=np.float32))[None], training=False).numpy()[0]

                    np.testing.assert_allclose(p, spec['expected_probabilities'], rtol=0.0001, atol=1e-05)

                    models[key] = model

            st.session_state['models'] = models

            st.session_state['model_signature'] = signature

        models = st.session_state['models']

    except Exception as exc:

        st.error(f'Cannot load prediction models: {exc}')

        st.stop()
        
        mode = st.radio(
        'Input',
        ['Image upload', 'Camera snapshot'],
        horizontal=True,
        key='prediction_input_mode'
        )

        if mode == 'Camera snapshot':
            upload = st.camera_input(
                'Driver portrait',
                key='prediction_camera'
            )
        else:
            upload = st.file_uploader(
                'Driver portrait',
                type=['jpg', 'jpeg', 'png', 'webp'],
                key='prediction_upload'
            )

    # mode = st.radio('Input', ['Image upload', 'Camera snapshot'], horizontal=True, key='prediction_input_mode')

    # if mode == 'Camera snapshot':

    #     upload = st.camera_input('Driver portrait', key='prediction_camera')

    # else:

    #     upload = st.file_uploader('Driver portrait', type=['jpg', 'jpeg', 'png', 'webp'], key='prediction_upload')
    
    
    if upload is not None:

        try:

            with Image.open(upload) as opened:

                portrait = opened.convert('RGB')

            st.image(portrait, width=600)

            if st.button('Predict image', key='prediction_submit'):

                width, height = portrait.size

                if width < 16 or height < 16:

                    raise ValueError('Please supply an image at least 16 pixels wide and high.')

                gray = cv2.cvtColor(np.asarray(portrait.convert('RGB')), cv2.COLOR_RGB2GRAY)

                gray = cv2.equalizeHist(gray)

                base = Path(cv2.data.haarcascades)

                face_detector = cv2.CascadeClassifier(str(base / 'haarcascade_frontalface_default.xml'))

                eye_detector = cv2.CascadeClassifier(str(base / 'haarcascade_eye_tree_eyeglasses.xml'))

                if face_detector.empty() or eye_detector.empty():

                    raise ValueError('OpenCV face/eye detector files are unavailable.')

                faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

                if len(faces):

                    x, y, w, h = map(int, max(faces, key=lambda rect: rect[2] * rect[3]))

                    upper = gray[y:y + round(0.6 * h), x:x + w]

                    eyes = eye_detector.detectMultiScale(upper, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))

                    candidates = [tuple(map(int, e)) for e in eyes if 0.12 * h <= e[1] + e[3] / 2 <= 0.52 * h and e[2] <= 0.5 * w]

                    if candidates:

                        ex, ey, ew, eh = min(candidates, key=lambda e: (e[0], -e[2] * e[3]))

                        box = (x + ex - 0.1 * ew, y + ey - 0.1 * eh, x + ex + 1.1 * ew, y + ey + 1.1 * eh)

                    else:

                        box = (x + 0.12 * w, y + 0.22 * h, x + 0.48 * w, y + 0.48 * h)

                else:

                    box = (0.35 * width, 0.25 * height, 0.55 * width, 0.4 * height)

                left, top, right, bottom = map(lambda value: int(round(value)), box)

                left, top = (max(0, min(left, width - 1)), max(0, min(top, height - 1)))

                right, bottom = (max(left + 1, min(right, width)), max(top + 1, min(bottom, height)))

                eye_input = portrait.crop((left, top, right, bottom))

                result = {}

                task_inputs = [('eyes', eye_input), ('yawn', portrait)]

                for task, image in task_inputs:

                    model = models[task + '_' + kind]

                    p = model(tf.keras.applications.mobilenet_v2.preprocess_input(np.asarray(image.convert('RGB').resize((224,224), Image.Resampling.NEAREST), dtype=np.float32))[None], training=False).numpy()[0]

                    if not np.isfinite(p).all() or (p < 0).any() or (not np.isclose(p.sum(), 1, atol=0.0001)):

                        raise ValueError(f'{task}: invalid model scores')

                    index = int(p.argmax())

                    score = float(p[index])

                    classes = metadata['models'][task][kind]['classes']

                    result[task] = classes[index] if score >= threshold else 'Uncertain'

                    result[task + '_confidence'] = score

                    result[task + '_top_class'] = classes[index]

                st.write('Eye state:', result['eyes'])

                labels = {'no_yawn': 'No yawn', 'yawn': 'Yawn', 'Uncertain': 'Uncertain'}

                st.write('Yawn state:', labels[result['yawn']])

                if result["eyes"] == "Closed":

                    fatigue_level = "Severe Fatigue"

                elif result["eyes"] == "Open" and result["yawn"] == "yawn":

                    fatigue_level = "Mild Fatigue"

                elif result["eyes"] == "Open" and result["yawn"] == "no_yawn":

                    fatigue_level = "Alert"

                else:

                    fatigue_level = "Uncertain"

                st.write("Fatigue level:", fatigue_level)

                with st.expander('View confidence'):

                    st.metric('Eye confidence', f"{result['eyes_confidence']:.1%}")

                    st.metric('Yawn confidence', f"{result['yawn_confidence']:.1%}")

                # st.caption("Rule-based stage from one image; sustained fatigue is not measured.")

                st.caption(

                    f"Below {threshold:.0%} is shown as Uncertain. "

                    "These are model scores, not measured correctness probabilities."

                )

        except Exception as exc:

            st.error(str(exc))

    


