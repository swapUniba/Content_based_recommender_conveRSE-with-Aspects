# Installazione

Per poter utilizzare il sistema è necessario andare ad installare le seguenti librerie:

- Gensim
- Flask
- SciPy
- Flask
- Numpy


# Contenuto del repository

- **Main.py**: interfaccia del web service. Gli endpoint definiti sono
    - */SelectModel/id*: gli id disponibili sono 1-7 per la selezione dei modelli
    - */getSuggestions*: prende in input l'oggetto JSON con i seguenti campi:
    ```JSON
        {
            movies
            entities
            movietoIgnore
            negativeEntity
            prefAspects
            negAspects
            recListSize
        }
    ```
    
- **RSCCore.py**: contiene le funzioni dei file che accedono alla Knowledge Base ed effettua i calcoli di similarità legati alle proprietà

- **Dataset**: contiene il file *MovieInfo.csv* ovvero il dataset dei film con annesse proprietà

- **Models**: ogni file identificato da *nome_modello*.py e seguono la stessa struttura. Inoltre, in ogni cartella devono essere presenti i file dei modelli già addestrati. Qualora non fossero presenti, i modelli saranno riaddestrati alla prima esecuzione del programma oppure sono scaricabili da https://drive.google.com/file/d/18aksc5-er653KZcLHBpunAh-Cibuv8X7/view?usp=sharing