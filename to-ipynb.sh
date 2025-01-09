#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: to-ipynb.sh <py file>"
    echo "Converts py file to ipynb."
    exit 1
fi
pyfile=$1

echo "Converting $pyfile to "$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"..."

url=https://colab.research.google.com/github/jwcalder/LAMLPython/blob/main/"$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"
qr_name="$(dirname "$pyfile")_$(basename "$pyfile" .py).png"
echo Creating QR code for $url and storing in QRCodes/$qr_name
qrencode -o QRCodes/$qr_name $url

sed 's/#pip install/pip install/' $pyfile | sed '/plt.ion/d' > temp.py
ipynb-py-convert temp.py "$(dirname "$pyfile")/$(basename "$pyfile" .py).ipynb"
rm temp.py

