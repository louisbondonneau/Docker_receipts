#!/bin/bash

# Définition du répertoire de destination
DEST="/usr/share/casacore/data"

# Vérification de l'existence du répertoire de destination
if [ ! -d "$DEST" ]; then
    echo "Le répertoire $DEST n'existe pas, création..."
    mkdir -p "$DEST"
fi

# Téléchargement des fichiers depuis le serveur FTP
wget -N -m --ftp-user=anonymous --ftp-password=anonymous -P "$DEST" ftp://ftp.astron.nl/outgoing/Measures/*


mv $DEST/ftp.astron.nl/outgoing/Measures/*.ztar $DEST
rm -r $DEST/ftp.astron.nl

cd $DEST
for i in *.ztar; do
    tar -xzvf "$i" || echo "Error extracting $i, possibly corrupted file."
done
