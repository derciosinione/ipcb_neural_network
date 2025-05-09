"""
Autores:
    Dércio Simione Bernardo Domingos (nº 20220103)
    Joel Dialamicua (nº 20221985)

Curso:
    Licenciatura em Engenharia Informática

Turma:
    Turma 1,
    Turma 2,
"""

import random
import math
import matplotlib.pyplot as plt
import csv

alpha = 0.2
student_class_position = 10

max_age = 0
max_academic_pressure = 0
max_study_satisfaction = 0
max_study_hours = 0
max_financial_stress = 0


# ------------------CÓDIGO GENÉRICO PARA CRIAR, TREINAR E USAR UMA REDE COM UMA CAMADA ESCONDIDA------------
def make(nx, nz, ny):
    """Funcao que cria, inicializa e devolve uma rede neuronal, incluindo
    a criacao das diversos listas, bem como a inicializacao das listas de pesos. 
    Note-se que sao incluidas duas unidades extra, uma de entrada e outra escondida, 
    mais os respectivos pesos, para lidar com os tresholds; note-se tambem que, 
    tal como foi discutido na teorica, as saidas destas unidades estao sempre a -1.
    por exemplo, a chamada make(3, 5, 2) cria e devolve uma rede 3x5x2"""
    # a rede neuronal é num dicionario com as seguintes chaves:
    # nx     numero de entradas
    # nz     numero de unidades escondidas
    # ny     numero de saidas
    # x      lista de armazenamento dos valores de entrada
    # z      array de armazenamento dos valores de activacao das unidades escondidas
    # y      array de armazenamento dos valores de activacao das saidas
    # wzx    array de pesos entre a camada de entrada e a camada escondida
    # wyz    array de pesos entre a camada escondida e a camada de saida
    # dz     array de erros das unidades escondidas
    # dy     array de erros das unidades de saida    

    nn = {'nx': nx, 'nz': nz, 'ny': ny, 'x': [], 'z': [], 'y': [], 'wzx': [], 'wyz': [], 'dz': [], 'dy': []}

    nn['wzx'] = [[random.uniform(-0.5, 0.5) for _ in range(nn['nx'] + 1)] for _ in range(nn['nz'])]
    nn['wyz'] = [[random.uniform(-0.5, 0.5) for _ in range(nn['nz'] + 1)] for _ in range(nn['ny'])]

    return nn


def sig(inp):
    """Funcao de activacao (sigmoide)"""
    return 1.0 / (1.0 + math.exp(-inp))


def forward(nn, input):
    """Função que recebe uma rede nn e um padrao de entrada input (uma lista) 
    e faz a propagacao da informacao para a frente ate as saidas"""
    # copia a informacao do vector de entrada input para a listavector de inputs da rede nn
    nn['x'] = input.copy()
    # adiciona a entrada a -1 que vai permitir a aprendizagem dos limiares
    nn['x'].append(-1)
    # calcula a activacao da unidades escondidas
    for i in range(nn['nz']):
        nn['z'] = [sig(sum([x * w for x, w in zip(nn['x'], nn['wzx'][i])])) for i in range(nn['nz'])]
        # adiciona a entrada a -1 que vai permitir a aprendizagem dos limiares
        nn['z'].append(-1)
        # calcula a activacao da unidades de saida
        nn['y'] = [sig(sum([z * w for z, w in zip(nn['z'], nn['wyz'][i])])) for i in range(nn['ny'])]


def error(nn, output):
    """Funcao que recebe uma rede nn com as activacoes calculadas
       e a lista output de saidas pretendidas e calcula os erros
       na camada escondida e na camada de saida"""
    nn['dy'] = [y * (1 - y) * (o - y) for y, o in zip(nn['y'], output)]
    zerror = [sum([nn['wyz'][i][j] * nn['dy'][i] for i in range(nn['ny'])]) for j in range(nn['nz'])]
    nn['dz'] = [z * (1 - z) * e for z, e in zip(nn['z'], zerror)]


def update(nn):
    """funcao que recebe uma rede com as activacoes e erros calculados e
    actualiza as listas de pesos"""

    nn['wzx'] = [[w + x * nn['dz'][i] * alpha for w, x in zip(nn['wzx'][i], nn['x'])] for i in range(nn['nz'])]
    nn['wyz'] = [[w + z * nn['dy'][i] * alpha for w, z in zip(nn['wyz'][i], nn['z'])] for i in range(nn['ny'])]


def iterate(i, nn, input, output):
    """Funcao que realiza uma iteracao de treino para um dado padrao de entrada input
    com saida desejada output"""
    forward(nn, input)
    error(nn, output)
    update(nn)
    # print('%03i: %s -----> %s : %s' %(i, input, output, nn['y']))


# -----------------------CÓDIGO QUE PERMITE CRIAR E TREINAR REDES PARA APRENDER AS FUNÇÕES BOOLENAS--------------------
"""Funcao que cria uma rede 2x2x1 e treina a função lógica AND
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""


def train_and(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [0])
        iterate(i, net, [1, 0], [0])
        iterate(i, net, [1, 1], [1])
    return net


"""Funcao que cria uma rede 2x2x1 e treina um OR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""


def train_or(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [1])
    return net


"""Funcao que cria uma rede 2x2x1 e treina um XOR
A função recebe como entrada o número de épocas com que se pretende treinar a rede"""


def train_xor(epocas):
    net = make(2, 2, 1)
    for i in range(epocas):
        iterate(i, net, [0, 0], [0])
        iterate(i, net, [0, 1], [1])
        iterate(i, net, [1, 0], [1])
        iterate(i, net, [1, 1], [0])
    return net


## ------------------------- MY CODE ------------------------- ##
# -------------------------CÓDIGO QUE IRÁ PERMITIR CRIAR UMA REDE PARA APRENDER A CLASSIFICAR ESTUDANTES---------

"""Funcao principal do nosso programa para prever situações de depressão em estudantes:
cria os conjuntos de treino e teste, chama a funcao que cria e treina a rede e, por fim, 
a funcao que a testa. A funcao recebe como argumento o ficheiro correspondente ao dataset 
que deve ser usado, os tamanhos das camadas de entrada, escondida e saída,
o numero de epocas que deve ser considerado no treino e os tamanhos dos conjuntos de treino e 
teste"""


def run_students(file: str, input_size: int, hidden_size: int, output_size: int, epochs: int, training_set_size: int,
                 test_set_size: int):
    training_set, test_set = build_sets(file, training_set_size, test_set_size)
    net = train_students(input_size, hidden_size, output_size, training_set, test_set, epochs)
    test_students(net, test_set, printing=False)


def read_file(nome_f):
    """
        Lê um arquivo CSV e retorna seus dados, ignorando a primeira linha.

        Parâmetros:
        nome_f (str): Caminho do arquivo CSV.

        Retorna:
        list: Dados do arquivo CSV (sem a primeira linha).
    """

    with open(nome_f, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        data = list(reader)
    return data


"""Funcao que cria os conjuntos de treino e de teste a partir dos dados
armazenados em f ('Depression Student Dataset.csv'). A funcao le cada linha, 
tranforma-a numa lista de valores e chama a funcao translate para a colocar no 
formato adequado para o padrao de treino. Estes padroes são colocados numa lista.
A função recebe como argumentos o nº de exemplos que devem ser considerados no conjunto de 
treino --->x e o nº de exemplos que devem ser considerados no conjunto de teste ------> y
Finalmente, devolve duas listas, uma com x padroes (conjunto de treino)
e a segunda com y padrões (conjunto de teste). Atenção que x+y não pode ultrapassar o nº 
de estudantes disponível no dataset"""


def build_sets(nome_f, x, y):
    data = read_file(nome_f)

    define_max_values(data)

    translated_data = [translate(row) for row in data]

    random.shuffle(translated_data)

    training_set = translated_data[:x]
    test_set = translated_data[x:x + y]

    return training_set, test_set


"""A função translate recebe cada lista de valores que caracterizam um estudante
e transforma-a num padrão de treino. Cada padrão é uma lista com o seguinte formato 
[padrao_de_entrada, classe_do_estudante, padrao_de_saida]
O enunciado do trabalho explica de que forma deve ser obtido o padrão de entrada
"""


def translate(lista: list) -> list:
    categoricos = converte_categ_numerico(lista)
    numericos = normaliza_valores(lista)

    input_pattern = categoricos + numericos

    student_class = lista[student_class_position]

    student_class_output_dict = {'Yes': [1, 0], 'No': [0, 1]}

    output_pattern = student_class_output_dict[student_class]

    return [input_pattern, student_class, output_pattern]


def converte_categ_numerico(instancia: list) -> list:
    """
    Converte valores categóricos em representações numéricas usando one-hot encoding.
    
    Parâmetros:
    - instancia: Uma lista contendo os valores categóricos de um estudante.
    
    Retorna:
    - Uma lista com os valores numéricos correspondentes.
    """

    gender = {
        'Male': [1, 0],
        'Female': [0, 1]
    }

    sleep_duration = {
        'Less than 5 hours': [1, 0, 0, 0],
        '5-6 hours': [0, 1, 0, 0],
        '7-8 hours': [0, 0, 1, 0],
        'More than 8 hours': [0, 0, 0, 1]
    }

    dietary_habits = {
        'Healthy': [1, 0, 0],
        'Moderate': [0, 1, 0],
        'Unhealthy': [0, 0, 1]
    }

    yes_no = {
        'Yes': [1, 0],
        'No': [0, 1]
    }

    gender_encoded = gender[instancia[0]]
    sleep_encoded = sleep_duration[instancia[4]]
    dietary_encoded = dietary_habits[instancia[5]]
    suicidal_encoded = yes_no[instancia[6]]
    family_history_encoded = yes_no[instancia[9]]

    return gender_encoded + sleep_encoded + dietary_encoded + suicidal_encoded + family_history_encoded


def normaliza_valores(lista: list) -> list:
    """
    Normaliza os valores numéricos para que estejam na mesma escala (entre 0 e 1).
    
    Parâmetros:
    - lista: Uma lista contendo os valores numéricos de um estudante.
    
    Retorna:
    - Uma lista com os valores normalizados.
    """

    age = float(lista[1]) / 100
    academic_pressure = float(lista[2]) / max_academic_pressure
    study_satisfaction = float(lista[3]) / max_study_satisfaction
    study_hours = float(lista[7]) / max_study_hours
    financial_stress = float(lista[8]) / max_financial_stress

    return [age, academic_pressure, study_satisfaction, study_hours, financial_stress]


"""Cria a rede e chama a funçao iterate para a treinar. A função recebe como argumento 
o conjunto de treino, os tamanhos das camadas de entrada, escondida e saída e o número 
de épocas que irão ser usadas para fazer o treino"""


def train_students(input_size: int, hidden_size: int, output_size: int, training_set: list, test_set: list,
                   epochs: int):
    network = make(input_size, hidden_size, output_size)

    training_list = list()
    test_list = list()

    for epoch in range(epochs):
        for item in training_set:
            iterate(epoch, network, item[0], item[2])

        train_accuracy = test_students(network, training_set, printing=False)
        test_accuracy = test_students(network, test_set, printing=False)

        training_list.append(train_accuracy)
        test_list.append(test_accuracy)

        print(f'Epoch {epoch + 1}: Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%')

    create_plot(epochs, training_list, test_list)

    return network


"""Funcao que avalia a precisao da rede treinada, utilizando o conjunto de teste ou treino.
Para cada padrao do conjunto chama a funcao forward e determina a classe do estudante
que corresponde ao maior valor da lista de saida. A classe determinada pela rede deve ser comparada com a classe real,
sendo contabilizado o número de respostas corretas. A função calcula a percentagem de respostas corretas"""


def test_students(net, test_set, printing=True):
    total_correct = 0

    for i, data in enumerate(test_set):
        forward(net, data[0])
        predicted_class = retranslate(net['y'])
        true_class = data[1]

        if predicted_class == true_class:
            total_correct += 1

        if printing:
            print(
                f'The network thinks student number {i + 1} has depression: {predicted_class}, it should be {true_class}')

    accuracy = (total_correct / len(test_set)) * 100

    if printing:
        print(f'Success rate: {accuracy:.2f}%')

    return accuracy


"""Recebe o padrao de saida da rede e devolve o estado de saúde do estudante.
O estado de saúde do estudante corresponde ao indice da saida com maior valor."""


def retranslate(out):
    return 'Yes' if out[0] > out[1] else 'No'


def define_max_values(data: list):
    """
      Calcula os valores máximos para os atributos numéricos do dataset e armazena em variáveis globais.

      Parâmetros:
      -----------
      data: list
          Lista de listas contendo os dados dos estudantes. Cada sublista representa um estudante.

      Variáveis Globais Definidas:
      ----------------------------
      max_age : float
          Valor máximo da idade (Age).
      max_academic_pressure : float
          Valor máximo da pressão acadêmica (Academic Pressure).
      max_study_satisfaction : float
          Valor máximo da satisfação com os estudos (Study Satisfaction).
      max_study_hours : float
          Valor máximo das horas de estudo (Study Hours).
      max_financial_stress : float
          Valor máximo do estresse financeiro (Financial Stress).
      """

    global max_age, max_academic_pressure, max_study_satisfaction, max_study_hours, max_financial_stress

    max_age = max(float(row[1]) for row in data)
    max_academic_pressure = max(float(row[2]) for row in data)
    max_study_satisfaction = max(float(row[3]) for row in data)
    max_study_hours = max(float(row[7]) for row in data)
    max_financial_stress = max(float(row[8]) for row in data)


def create_plot(epochs, train_accuracies, test_accuracies):
    """
    Plota um gráfico de acurácia de treinamento e teste ao longo das épocas.

    Args:
        epochs (int): Número total de épocas (iterações de treinamento).
        train_accuracies (list[float]): Acurácias no conjunto de treinamento por época.
        test_accuracies (list[float]): Acurácias no conjunto de teste por época.

    Exemplo:
        create_plot(5, [0.75, 0.80, 0.85, 0.88, 0.90], [0.70, 0.76, 0.80, 0.84, 0.85])
    """

    plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy over Epochs')
    plt.legend(loc='lower right')
    plt.show()


# if __name__ == "__main__":
#     #Vamos treinar durante 1000 épocas uma rede para aprender a função logica AND
#     #Faz testes para números de épocas diferentes e para as restantes funções lógicas já implementadas
#     rede_AND = train_and(1000)
#     #Agora vamos ver se ela aprendeu bem
#     tabela_verdade = {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}
#     for linha in tabela_verdade:
#         forward(rede_AND, list(linha))
#         print('A rede determinou %s para a entrada %d AND %d quando devia ser %d'%(rede_AND['y'], linha[0], linha[1], tabela_verdade[linha]))

if __name__ == "__main__":
    input_size, hidden_size, output_size = 18, 11, 2
    epochs = 5
    training_set_size = 379
    test_set_size = 123
    file = 'Depression Student Dataset.csv'
    run_students(file, input_size, hidden_size, output_size, epochs, training_set_size, test_set_size)
