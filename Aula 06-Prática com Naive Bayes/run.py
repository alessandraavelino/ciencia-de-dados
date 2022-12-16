from naivebayesclassifier import *
from fileinput import filename
import glob
from typing import List

# Vamos usar a linha de assunto na identificação do spam

# modifique o caminho para indicar o local dos arquivos
path = 'spam_data/*/*/*'

data: List[Message] = []

# glob.glob retorna todos os nomes de arquivos que correspondem ao caminho com curinga
for filename in glob.glob(path):
    is_spam = "ham" not in filename

    # Existem alguns caracteres de lixo nos e-mails;
    # o errors='ignore' os ignora em vez de gerar uma exceção.
    with open(filename, errors='ignore') as email_file:
        for line in email_file:
            if line.startswith("Subject:"):
                subject = line.lstrip("Subject: ")
                data.append(Message(subject, is_spam))
                break # arquivo finalizado

# Dividimos o conjunto de dados em dados de treinamento
# e de teste para usar o classificador

import random
from scratch.machine_learning import split_data

random.seed(0)  # Para que você chegue aos mesmos resultados que o autor do livro
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)

# Classificando algumas mensagens como spam para verificar o funcionamento do modelo
from collections import Counter

predictions = [(message, model.predict(message.text))
                for message in test_messages]

# Presuma que spam_probability > 0.5 corresponde à previsão de spam
# e conte as combinações de (real is_spam, previsto is_spam)
confusion_matrix = Counter((message.is_spam, spam_probability > 0.5)
                            for message, spam_probability in predictions)

print(confusion_matrix)

# Inspecionando o interior do modelo para determinar as palavras menos e
# mais indicativas de spam:

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    # Aqui, não recomento chamar métodos privados, mas é por uma boa causa.
    prob_if_spam, prob_if_ham = model._probabilities(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)

words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])