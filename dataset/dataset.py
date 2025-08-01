import pandas as pd

def get_dataset(path : str = './dataset/dataset.pkl') -> pd.DataFrame:
    '''
    Returns the dataset containing ```gesture_id```\'s and the sequences of ```Mediapipe```\'s landmarks.

    #### Columns:

    - ```gesture_id```: int, zero-indexed id of a gesture,
    - ```lmk_seq```: numpy array of shape (n_frames, 63),   
    where n_frames is the number of frames of the gesture sample and 63 is the amount of features: each of the 21 landmarks described by the 3 axes (x,y,z)
    '''
    dataset = pd.read_pickle(path)
    # Enforce zero-indexing
    dataset['gesture_id'] -= 1
    return dataset

gestures = [
"Non-gesture",
"Pointing with one finger",
"Pointing with two fingers",
"Click with one finger",
"Click with two fingers",
"Throw up",
"Throw down",
"Throw left",
"Throw right",
"Open twice",
"Double click with one finger",
"Double click with two fingers",
"Zoom in",
"Zoom out",
]