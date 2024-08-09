import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from datetime import datetime
import pytz
from flask import Flask, render_template, request, jsonify, send_file

#Metodo para guardar los nombres de las personas reconocidas
#con su respectiva fecha y hora
def markAttendance(name):
    # Abrimos el archivo en modo lectura y escritura
    with open('/home/CapitanNemo/mysite/data.csv', 'r+') as h:
        # Leemos la informacion
        data = h.readline()
        # Creamos lista de nombres
        listanombres = []

        # Iteramos cada linea del doc
        for line in data:
            # Buscamos la entrada y la diferencias con,
            entrada = line.split(',')
            # Almacenamos los nombres
            listanombres.append(entrada[0])

        # Verificamos si ya hemos almacenado el nombre
        if name not in listanombres:
            # Obtener la zona horaria de Mexico
            mexico_timezone = pytz.timezone('America/Mexico_City')
            # Extraemos informacion actual
            info = datetime.now(mexico_timezone)
            # Extraemos fecha
            fecha = info.strftime("%d/%m/%Y")
            # Extraemos hora
            hora =info.strftime("%I:%M:%S %p")

            # Guardamos la informacion
            h.writelines(f'{name}, {fecha}, {hora}\n')
            print(info)


#Metodo que identifica las similitudes entre rostros conocidos y no conocidos
def similarity_percentage(known_encodings, unknown_encoding, names):
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    similarities = (1 - distances / np.sqrt(2)) * 100
    similar_indices = similarities >= 34.0

    return similarities[similar_indices], names[similar_indices]

app = Flask(__name__)

@app.route('/')
def descargar_csv():
    # Ruta relativa al archivo CSV en el directorio de tu aplicación Flask
    csv_file = 'data.csv'
    return send_file(csv_file, as_attachment=True)

@app.route('/detectar_rostro', methods=['POST'])
def detectar_rostro():
    if 'image' not in request.files:
        return jsonify({'mensaje': 'No se encontró una imagen en la solicitud'})
    archivo = request.files['image']
    if archivo.filename == '':
        return jsonify({'mensaje': 'No se proporcionó un nombre de archivo.'})
    nombre_archivo = archivo.filename
    app.config['UPLOAD_FOLDER'] = '/home/CapitanNemo/mysite/uploads'
    archivo_guardado = os.path.join(app.config['UPLOAD_FOLDER'], nombre_archivo)
    archivo.save(archivo_guardado)

    imagen = cv.imread(archivo_guardado)
    if imagen is not None and imagen.shape[0] > 0 and imagen.shape[1] > 0:
        #INITIALIZE
        facenet = FaceNet()
        faces_embeddings = np.load('/home/CapitanNemo/mysite/face_embeddings_done4classes.npz')
        Y = faces_embeddings['arr_1']
        nose = faces_embeddings['arr_0']
        encoder = LabelEncoder()
        encoder.fit(Y)
        haarcascade = cv.CascadeClassifier('/home/CapitanNemo/mysite/haarcascade_frontalface_default.xml')
        model = pickle.load(open('/home/CapitanNemo/mysite/svm_model_160x160.pkl', 'rb'))

        #se carga la imagen y se convierte a escala de grises
        image = cv.imread(archivo_guardado)
        rgb_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        faces = haarcascade.detectMultiScale(gray_img, 1.1, 5)
        counts = {}

        for (x, y, w, h) in faces:
            img = rgb_img[y:y+h, x:x+w]
            img = cv.resize(img, (160,160)) # 1x160x160x3
            img = np.expand_dims(img,axis=0)

            ypred = facenet.embeddings(img)
            face_name = model.predict(ypred)

            final_name = encoder.inverse_transform(face_name)[0]

            similarities, similar_names = similarity_percentage(nose, ypred, Y)
            print("Porcentajes de similitud:", similarities)

            counts[final_name] = len(similarities)
            print("Número de coincidencias:", counts[final_name])
            if counts[final_name] > 0:
                print("-->", final_name)
                print("Nombres similares:", similar_names)
                cv.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 2)
                cv.putText(image, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,0,255), 2, cv.LINE_AA)
            else:
                final_name = "desconocido"
                cv.rectangle(image, (x,y), (x+w,y+h), (255,0,255), 2)
                cv.putText(image, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                           0.9, (0,255,255), 2, cv.LINE_AA)

            markAttendance(final_name)


        return jsonify({'mensaje': 'El rostro ha sido detectado correctamente', 'imagen': nombre_archivo})
        #image = open(archivo_guardado, 'rb')
        #return send_file(image, mimetype='image/png')
    else:
        return jsonify({'mensaje': 'La imagen no se ha cargado correctamente o esta vacia'})


if __name__ == '__main__':
    app.run(debug=True)