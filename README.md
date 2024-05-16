# Keep On Gambling
### Projekt nr. 1 na przedmiot inteligencja obliczeniowa
##### 4 semestr Informatyki Praktycznej UG
---

### Opis aplikacji
Aplikacja webowa służąca do klasyfikacji kart do gry na podstawie zdjęcia zrobionego kamerką.
Do wyboru są 3 warianty:
- KOG (Keep On Gambling) - model w pełni wytrenowany przeze mnie przy użyciu biblioteki TensorFlow.
- Model YOLO (You Only Look Once) - model, który bazuje na gotowym rozwiązaniu rozpoznawania obiektów w czasie rzeczywistym *Ultralytics YOLOv8*. Model ten został wytrenowany do wykrywania samych kart na tym samym datasecie co model KOG, metodą tzw. transer learning (fine-tuning).
- YOLO + KOG - Połączenie funckjonalne obu modeli. Model YOLO wykorzystujemy tutaj **jedynie** do wykrywania na jakim obszarze znajduje się karta. Ta karta jest następnie wycinana i podawana modelowi KOG do oceny.

Wariantom 2 i 3 można edytować confidence (stopień pewności) z poziomu aplikacji.

### Uruchomienie aplikacji

Aby uruchomić aplikację należy uruchomić 2 serwisy
#### Serwer (Python)
Część serwerowa (oraz przykłady procesu trenowania) bazują na interpreterze Python. Żeby móc skorzystać z aplikacji na początku należy pobrać zależności za pomoca polecenia:
```bash
pip install -r requirements.txt
```

Następnie w katalogu `predict_server` uruchamiamy plik `app.py` za pomoca polecenia:
```bash
python3 app.py
```

#### Klient (JavaScript)
Część klienta bazuje na JavaScriptowym frameworku Next.js. Zanim uruchomimy aplikację, tak samo jak w przypadku Pythona, musimy pobrać zależności. Wykonujemy to w katalogu `keep-on-gambling` za pomocą polecenia:
```bash
npm install
```

Następnie w tym samym katalogu `keep-on-gambling` uruchamiamy aplikację za pomocą polecenia
```bash
npm run dev
```

### Proces treningowy

W katalogu `training_process` znajdują się wszystkie pliki, które posłużyły mi przy trenowaniu zarówno własnego modelu CNN `KOG` oraz fine-tuningu modelu `YOLOv8`.

Dataset użyty przy trenowaniu obu modeli pochodzi stąd:
https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification

#### Model KOG

Podczas trenowania modelu, przeprowadziłem wiele iteracji, dostosowując różne parametry sieci, takie jak liczba warstw, liczba neuronów w każdej warstwie, funkcje aktywacji, algorytm optymalizacji, tempo uczenia się i inne. Każda zmiana tych parametrów wpłynęła na to, jak dobrze model uczył się z danych treningowych i jak dobrze generalizował nowe dane.

**Augmentacja danych** to technika stosowana do sztucznego zwiększenia ilości danych treningowych poprzez wprowadzanie małych zmian do istniejących próbek. Może to obejmować operacje takie jak obracanie, skalowanie, przesuwanie, odbijanie lustrzane i wiele innych. W przypadku kart próbowałem dodać augmentację danych w postaci zniekształcania, obracania i zwiększania kontrastu - co okazało się słabym pomysłem, gdyż przez takie przesunięcia, zdjęcia kart były przycięte, co negatywnie wpływała na prawidłowe uczenie się sieci. W następnych kilku rewizjach, próbowałem zaradzić temu problemowi poprzez stosowanie odpowiedniego padding (obramówki wokół zdjęć). Zacząłem od prostej białej i czarnej obramówki, ale precyzja modelu nie rosła. Idąc myślą, że być może model uczy się schematów ów paddingu, zacząłem stostowac padding różnokolorowy z podobnym skutkiem. Ostateczną próbę zastosowania połączenia paddingu + augmentacji zastosowałem poprzez nadanie paddingu w formie pomieszanych pikseli różnego koloru, co miało zapobiec jakimkolwiek skojarzaniem kart z tłem. Ku mojemu zdziwieniu niestety nawet i ta zmiana nie pomogła. Po cofnięciu wszystkich zmian i pozostawieniu jedynie augmentacji w postaci `layers.GaussianNoise(0.01)` (patrz `model_experiments/model_8/model.py`), nauczyłem model KOG rozpoznawać karty (z bliska) na poziomie ~95%.

**Podsumowanie sieci, hiperparametry**
Ostateczna implementacja sieci konwolucyjnej KOG wygląda następująco:
```python
num_classes = len(class_names)
    model = Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1. / 255),
        layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(512, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dense(num_classes, name="outputs")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
```

Stworzona przeze mnie została konwolucyjna sieć neuronową (CNN) za pomocą TensorFlow i Keras.

- **Warstwy konwolucyjne (Conv2D):** Są to podstawowe bloki budowlane CNN. Filtry w tych warstwach przesuwają się przez obraz, przetwarzając małe kawałki na raz, co pozwala modelowi zrozumieć lokalne wzorce obrazu. Liczba filtrów w warstwie konwolucyjnej określa liczbę cech, które model może nauczyć się rozpoznawać.
- **Warstwy normalizacji wsadowej (BatchNormalization):** Te warstwy są używane do przyspieszenia uczenia się i stabilizacji procesu uczenia poprzez normalizację aktywacji wejściowych.
- **Warstwy MaxPooling2D:** Te warstwy są używane do redukcji wymiarowości, co pomaga zmniejszyć liczbę parametrów modelu i zapobiegać przetrenowaniu.
- **Warstwy Dropout:** Są one używane do zapobiegania przetrenowaniu poprzez losowe wyłączanie pewnej liczby neuronów podczas każdej iteracji treningu.
- **Warstwy Dense:** Są to standardowe warstwy sieci neuronowej, które uczą się globalnych wzorców w danych.

Wybór tych konkretnych parametrów i warstw jest wynikiem eksperymentów i doświadczeń. Każdy z tych elementów ma swoją rolę w modelu i wpływa na to, jak dobrze model uczy się z danych i jak dobrze generalizuje na nowe dane.

#### Model YOLO

Model ten domyślnie miał służyć jedynie jako narzędzie, które wykryje kartę na zdjęciu i ją wytnie. Ostatecznie jednak, ponieważ nie chciałem korzystać z gotowej wytrenowanej na kartach sieci, sam to uczyniłem. Żeby zrobić to skutecznie musiałem przerobić dataset z kartami na taki, który odpowiada formatowi ultralytics YOLO. Do tego posłużyły mi skrypty przerabiające strukturę plików (patrz różnice pomiędzy `partial_datasets/cards_kaggle`, a `partial_datasets/cards_kaggle_yolo_padded_v5`). Największym wyzwaniem jednak było takie przerobienie datasetu, żeby model nauczył się rozpoznawać **gdzie** karta znajduje się na zdjęciu, a nie jaka to jest karta (czego nauczenie wyszło bardzo szybko temu modelowi bez żadnej dodatkowej obróki danych). Żeby nauczyć model rozpoznawać gdzie znajdują się na obrazie karty, wróciłem do pomysłu augmentacji danych z sekcji KOG. Wykonałem liczne eksperymenty (stąd v5 w nazwie folderu), które można przestudiować w `yolo_models/data_prep_scripts_experiments`. Najcięższym zadaniem było dynamiczne dostosowanie tzw. `bounding box`, który miał wskazywać sieci gdzie znajduje się zdjęcie na obrazie po nadaniu 'rozmazanego' paddingu. Przy piątej głównej rewizji skryptu udało się to zrobić z prawie zerowym błędem. Pozwoliło to wytrenować model, który z dosyć wysoką dokładnością nie dość, że rozpoznaje jaka karta znajduje się na obrazku, to jeszcze w którym miejscu.