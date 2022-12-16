from io import BytesIO  # Agora podemos tratar bytes como um arquivo
import requests         # Para baixar os arquivos, que
import tarfile          # estão no formato .tar.bz.

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
FILES = [   "20021010_easy_ham.tar.bz2",
            "20021010_hard_ham.tar.bz2",
            "20021010_spam.tar.bz2",
            "20030228_easy_ham.tar.bz2",
            "20030228_hard_ham.tar.bz2",
            "20030228_spam.tar.bz2",
]

# Os dados ficarão aqui,
# nos subdiretórios /spam, /easy_ham e /hard_ham
# Altere para o diretório escolhido.
OUTPUT_DIR = 'spam_data'

for filename in FILES:
    # Use solicitações para obter o conteúdo dos arquivos em cada URL.
    content = requests.get(f"{BASE_URL}/{filename}").content

    # Encapsule os bytes na memóriapara usálos como um "arquivo"
    fin = BytesIO(content)

    # E extraia todos os arquivos para o diretório de saída especificado.
    with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
        tf.extractall(OUTPUT_DIR+"/"+filename)
