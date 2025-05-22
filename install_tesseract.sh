#!/bin/bash

# Installer les dépendances nécessaires
apt-get update && apt-get install -y \
    libleptonica-dev \
    libtesseract-dev \
    make \
    pkg-config \
    gcc \
    g++ \
    git \
    wget \
    unzip \
    tesseract-ocr

# Installer Tesseract 5.x depuis les sources
cd /tmp
git clone https://github.com/tesseract-ocr/tesseract.git
cd tesseract
git checkout 5.3.1
./autogen.sh
./configure
make -j$(nproc)
make install
ldconfig

# Ajouter les langues
tesseract --list-langs
wget https://github.com/tesseract-ocr/tessdata_fast/raw/main/fra.traineddata -P /usr/local/share/tessdata/
wget https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata -P /usr/local/share/tessdata/
