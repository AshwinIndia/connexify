
import mediapipe as mp 
import cv2
import numpy as np


mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 


cap = cv2.VideoCapture(0)


cv2.namedWindow('Raw Webcam Feed', cv2.WINDOW_NORMAL)


screen_res = (1920, 1080) 
cv2.setWindowProperty('Raw Webcam Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow('Raw Webcam Feed', screen_res)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        
        results = holistic.process(image)
        
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                                 )

        
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
                                 )

        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()



results.face_landmarks.landmark[0].visibility

import csv
import os
import numpy as np



num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
num_coords = len(results.pose_landmarks.landmark)
print(num_coords)
type(results.left_hand_landmarks)


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]



landmarks



#!!Do not run!!!!!!!!!!
with open('coords.csv', mode='w', newline='') as f:
    
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
    



#!!Do not run!!!!!!!!!!
class_name = "testing image"




cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        
        results = holistic.process(image)
        
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
       
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
       
        try:
         
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
           
            row=pose_row
            
            
            row.insert(0, class_name)
            
            
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


import pandas as pd
df = pd.read_csv('coords.csv')
df[df['class']==class_name]



# import pandas as pd
# import numpy as np
# from scipy.stats import zscore

# #Clean the data
# df = pd.read_csv('coords.csv')
# numeric_columns = df.select_dtypes(include=[np.number]).columns
# df.dropna(inplace=True)  # drop rows with missing values
# df = df[numeric_columns].apply(zscore)
# df = df[(np.abs(df) < 3).all(axis=1)]



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)
df_normalized = pd.DataFrame(df_normalized, columns=df.columns)
df_normalized['class'] = df['class']
df = df_normalized


import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('coords old.csv')

X = df.drop('class', axis=1) # features
y = df['class'] # target value
X
import pandas as pd
import numpy as np
from scipy.stats import zscore

# #Clean the data
# numeric_columns = X.select_dtypes(include=[np.number]).columns
# X.dropna(inplace=True)  # drop rows with missing values
# outlier_mask = ~(np.abs(X) < 3).all(axis=1)
# removed_rows = X[outlier_mask].index
# X= X[numeric_columns].apply(zscore)
# X= X[(np.abs(X) < 3).all(axis=1)]
# y = y.drop(removed_rows)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)



y_test



from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier(max_iter=1000, alpha=0.1)),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=2,
                                                                 random_state=42, max_depth=5, max_features='sqrt')),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=1000)),
}


fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model


fit_models



fit_models['rc'].predict(X_test)


from sklearn.metrics import accuracy_score
import pickle 



for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat) )
    

fit_models['rf'].predict(X_test)



with open('body lang.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)



with open('body lang.pkl', 'rb') as f:
    model = pickle.load(f)




cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
     
        results = holistic.process(image)
       
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
       
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
        #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                          )
        
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
                                 )

       
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
                                 )

        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                                 )
       
        try:
            
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            
            #row = pose_row+face_row
            row=pose_row
            

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            
            body_language_prob = model.predict_proba(X)[0]
          
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [1280,720]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            cv2.rectangle(image, (0,0), (250*3, 60), (245, 117, 16), -1)
            
            
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
           
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

