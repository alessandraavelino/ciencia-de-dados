from email.message import Message
from typing import List, Tuple, Dict, Iterable, Set, NamedTuple
import math, re
from collections import defaultdict

class Message(NamedTuple):
    text: str
    is_spam: bool

class NaiveBayesClassifier:
    def __init__(self, k:float = 0.5) -> None:
        self.k = k # fator de suavização

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict [str, int] = defaultdict(int)

        self.spam_messages = self.ham_messages = 0
    
    def train(self, messages: Iterable[Message]) -> None:
        # Incrementa a contagem de mensagens
        for message in messages:
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
    
            # Incremente as contagens de palavras
            for token in tokenize(message.text):
                self.tokens.add(token)
                if message.is_spam:
                    self.token_spam_counts[token] += 1
                else:
                    self.token_ham_counts[token] += 1
    
    def _probabilities(self, token:str) -> Tuple[float, float]:
        """retorna P(token | spam) e P(token | ham)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham  + self.k) / (self.ham_messages  + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Itere em cada palavra do vocabulário
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self._probabilities(token)
            # Se o *token* aparecer na mensagem,
            # adicione o log da chance de vê-lo
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham  += math.log(prob_if_ham)
            else:
                # Se não, adicione o log da probabilidade de _não_ vê-lo,
                # que corresponde a log(1-probabilidade de vê-lo)
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham  += math.log(1.0 - prob_if_ham)
        
        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham  = math.exp(log_prob_if_ham)
        
        return prob_if_spam / (prob_if_spam + prob_if_ham)

def tokenize(text: str) -> Set[str]:
    text = text.lower()                             # Converte para minúsculas.
    all_words = re.findall("[a-z0-9êé']+", text)    # extraia as palavras e
                                                    # remova as duplicadas
    return set(all_words)

# Vamos testar o modelo
if(__name__=="__main__"):
    assert tokenize("Ciência de dados é ciência") == {  'dados', 
                                                    'ciência',
                                                    'de',
                                                    'é'}

    messages = [Message("spam rules",   is_spam = True),
                Message("ham rules",    is_spam = False),
                Message("hello ham",    is_spam = False)]
    
    model = NaiveBayesClassifier(k=0.5)
    model.train(messages)

    # Verifica se as contagens estão corretas
    assert model.tokens == {"spam", "ham", "rules", "hello"}
    assert model.spam_messages == 1
    assert model.ham_messages  == 2
    assert model.token_spam_counts == {"spam":1,   "rules":1}
    assert model.token_ham_counts  == {"ham":2,    "rules":1,   "hello":1}

    # Fazer uma previsão de um texto para saber se ele é spam. Precisamos também
    # calcular manualmente os valores para comparar 

    text = "hello spam"

    probs_if_spam = [
        (1 + 0.5)     / (1 + 2*0.5),    # "spam"    (presente)
        1 - (0 + 0.5) / (1 + 2*0.5),    # "ham"     (ausente)
        1 - (1 + 0.5) / (1 + 2*0.5),    # "rules"   (ausente)
        (0 + 0.5)     / (1 + 2*0.5)     # "hello"   (presente)
    ]

    probs_if_ham = [
        (0 + 0.5) / (2 + 2 * 0.5),      # "spam"    (presente)
        1 - (2 + 0.5) / (2 + 2*0.5),    # "ham"     (ausente)
        1 - (1 + 0.5) / (2 + 2*0.5),    # "rules"   (ausente)
        (1 + 0.5) / (2 + 2 * 0.5),      # "hello"   (presente)
    ]

    p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
    p_if_ham  = math.exp(sum(math.log(p) for p in probs_if_ham))

    # Aproximadamente 0.8350
    assert int(10000*model.predict(text)) == int(10000*(p_if_spam / (p_if_spam + p_if_ham)))