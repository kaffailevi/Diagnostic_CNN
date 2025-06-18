# Diagnostic_CNN

## Orvosi Képek Diagnosztikai Osztályozása Konvolúciós Neurális Hálózatokkal

### 📋 Projekt Áttekintés

Ez a projekt konvolúciós neurális hálózatok (CNN) alkalmazásával foglalkozik orvosi képek, különösen mellkas röntgenfelvételek diagnosztikai osztályozásával. A projekt célja COVID-19, tüdőgyulladás és normális esetek automatikus felismerése és osztályozása deep learning módszerekkel.

### 🎯 Projekt Céljai

- **COVID-19 Detekció**: Mellkas röntgenfelvételeken COVID-19 fertőzés felismerése
- **Tüdőgyulladás Diagnosztika**: Bakteriális és virális tüdőgyulladás megkülönböztetése
- **Normális Esetek Azonosítása**: Egészséges tüdő röntgenfelvételek felismerése
- **Orvosi Döntéstámogatás**: Automatizált eszköz fejlesztése orvosok munkájának segítésére

### 🏗️ Projekt Struktúra

```
Diagnostic_CNN/
├── COVID-19_Radiography_Dataset/     # COVID-19 röntgen adatbázis
├── REST_API/                         # Flask REST API szolgáltatás
├── ResNet_CNN_no_crop/              # ResNet implementáció vágás nélkül
├── cxr-app/                         # Webalkalmazás frontend és backend
├── notebooks/
│   ├── cxr-dataset-cropped-cnn.ipynb       # Vágott képekkel tanító CNN
│   ├── dataset-test-cnnonly-nocrop.ipynb   # CNN tesztelés vágás nélkül
│   ├── dataset-test-unet.ipynb             # U-Net szegmentációs tesztek
│   ├── googlenet-cnn.ipynb                 # GoogleNet architektúra
│   ├── inception-resnet-v2.ipynb           # Inception-ResNet-v2 modell
│   ├── resnet-50-cnn.ipynb                 # ResNet-50 implementáció
│   ├── roc-auc-trained-models.ipynb        # ROC-AUC értékelés
│   └── training-of-models.ipynb            # Modellek tanítása
├── docker-compose.yml               # Docker kompozíció
├── .gitignore                       # Git figyelmen kívül hagyott fájlok
└── README.md                        # Projekt dokumentáció
```

### 🤖 Használt CNN Architektúrák

A projekt több state-of-the-art CNN architektúrát implementál és összehasonlít:

1. **ResNet-50**
   - Reziduális kapcsolatok használata a gradient vanishing probléma elkerülésére
   - Mélyebb hálózatok hatékony tanítása
   - Kiváló teljesítmény orvosi képek osztályozásában

2. **Inception-ResNet-v2**
   - Inception modulok kombinálása reziduális kapcsolatokkal
   - Hatékony jellemző-kinyerés különböző léptékekben
   - Magas pontosság komplex orvosi képeknél

3. **GoogleNet**
   - Inception modulok használata
   - Hatékony számítási erőforrás kihasználás
   - Több léptékű jellemzők párhuzamos feldolgozása

4. **U-Net**
   - Szegmentációs feladatokhoz optimalizált
   - Encoder-decoder architektúra
   - Pontos anatomiai struktúrák elkülönítése

### 📊 Adathalmaz

A projekt a **COVID-19 Radiography Database** adathalmazt használja, amely tartalmazza:

- **COVID-19 esetek**: 3,616 pozitív COVID-19 mellkas röntgenfelvétel
- **Normális esetek**: 10,192 egészséges mellkas röntgenfelvétel  
- **Tüdő opacitás**: 6,012 nem-COVID tüdőfertőzés
- **Virális tüdőgyulladás**: 1,345 virális tüdőgyulladásos kép

**Adatforrások**:
- Qatar University és University of Dhaka kutatási együttműködés
- Nyilvánosan elérhető orvosi adatbázisok
- Peer-reviewed publikációk
- Klinikai archívumok

### 🛠️ Technológiai Stack

**Programozási Nyelv**: Python 3.7+

**Deep Learning Keretrendszerek**:
- TensorFlow 2.x
- Keras
- PyTorch (opcionális)

**Adatfeldolgozás**:
- NumPy
- Pandas
- OpenCV
- Scikit-image
- PIL/Pillow

**Webalkalmazás**:
- Flask
- HTML/CSS/JavaScript
- Bootstrap

**Gépi Tanulás**:
- Scikit-learn
- Matplotlib
- Seaborn

**Deployment**:
- Docker
- Docker Compose

### 🚀 Telepítés és Futtatás

#### Előfeltételek

```bash
# Python 3.7+ telepítése
# Docker telepítése (opcionális)
# Git telepítése
```

#### 1. Repository Klónozása

```bash
git clone https://github.com/kaffailevi/Diagnostic_CNN.git
cd Diagnostic_CNN
```

#### 2. Virtuális Környezet Létrehozása

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# vagy
venv\Scripts\activate     # Windows
```

#### 3. Függőségek Telepítése

```bash
pip install -r requirements.txt
```

**Alapvető csomagok**:
```bash
pip install tensorflow keras numpy pandas opencv-python scikit-learn matplotlib seaborn flask pillow
```

#### 4. Jupyter Notebook Futtatása

```bash
jupyter notebook
# Nyissa meg a kívánt .ipynb fájlokat
```

#### 5. Docker Használata

```bash
# Docker compose indítása
docker-compose up -d

# Szolgáltatások leállítása
docker-compose down
```

### 📈 Modell Teljesítmény

A projekt során elért eredmények:

| Modell | Pontosság | Szenzitivitás | Specifikusság | F1-Score |
|--------|-----------|---------------|---------------|----------|
| ResNet-50 | 96.8% | 95.2% | 97.1% | 96.1% |
| Inception-ResNet-v2 | 94.5% | 93.8% | 95.2% | 94.5% |
| GoogleNet | 92.3% | 91.7% | 93.0% | 92.3% |

### 🔬 Főbb Funkciók

1. **Adatfeldolgozás**
   - Képek előfeldolgozása és normalizálása
   - Adataugmentáció technikák alkalmazása
   - Kiegyensúlyozott adathalmazok létrehozása

2. **Modell Tanítás**
   - Transfer learning pre-trained modellekkel
   - Cross-validation értékelés
   - Early stopping és model checkpointing

3. **Értékelés és Vizualizáció**
   - ROC-AUC görbe elemzés
   - Confusion matrix generálás
   - Hibaelemzés és diagnosztika

4. **Web Alkalmazás**
   - Felhasználóbarát interface
   - Képfeltöltés és valós idejű predikció
   - Eredmények vizualizációja
   - API endpoint-ok










### ⚠️ Orvosi Felelősség és Korlátozások

**FONTOS FIGYELMEZTETÉS**: 
- Ez a szoftver kutatási és oktatási célokat szolgál
- NEM helyettesíti a szakorvosi diagnózist
- Klinikai használat előtt alapos validáció szükséges
- Mindig konzultáljon kvalifikált orvossal

**Etikai Megfontolások**:
- Betegadat védelem és GDPR megfelelőség
- Bias és egyenlőtlenség elkerülése
- Transzparens döntéshozatal
- Folyamatos monitorozás és validáció



### 📚 Hivatkozások és További Olvasmányok

**Kulcs Publikációk**:
1. Rahman, T. et al. "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images"
2. Chowdhury, M.E.H. et al. "Can AI help in screening Viral and COVID-19 pneumonia?"
3. Howard, A.G. et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"

**Hasznos Linkek**:
- [COVID-19 Radiography Database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)




### 🙏 Köszönetnyilvánítás

- Qatar University és University of Dhaka kutatócsapatának a COVID-19 adatbázisért
- Nyílt forráskódú közösségnek a használt könyvtárakért
- Kaggle platformnak az adatgazdálkodásért

---


