# Arquivo: test_my_classifier.py

# Coloque a definição completa e corrigida da sua classe aqui
# (incluindo a linha _estimator_type = "classifier")
from src.gaussian_bayesian_classifier import GaussianBayesClassifier

from sklearn.utils.estimator_checks import check_estimator

print("Iniciando a verificação de compatibilidade do GaussianBayesClassifier...")

# Instancia o seu classificador
my_classifier = GaussianBayesClassifier()

# Roda a suíte completa de testes
check_estimator(my_classifier)

print("\nVerificação concluída com sucesso! Seu estimador é compatível com o Scikit-learn.")