## [Main repository](https://github.com/Elpiu/cnn-evaluate-parkinson-bd)

Applicazione web con flask deployata su heroku.

L'applicazione permtte di caricare un immagine e di farla classificare dal cnn.


Qui puoi trovare alcune immagini di esempio [QUI](https://github.com/Elpiu/flask-app-cnn-parkinson/tree/main/testImages).
Le cartelle rappresentano le categorie del classificatore.

### Cambiare modello

Per cambiare il modello basta sostituire il file '.h5' presente
nella cartella '/static/models'.

### Setting up Flask Env Python in visual Sudio Code

1. Crea una nuova cartella e arpila su VSC
2. Lancia un nuovo terminale
3. Comando: pip install virtualenv
4. Comando: virtualenv env
5. Comando: env\Scripts\activate
6. Comando: pip install flask
7. Crea app.py
8. Comando: flask run
