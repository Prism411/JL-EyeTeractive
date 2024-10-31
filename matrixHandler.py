import matplotlib.pyplot as plt
import math

# Dados dos pontos
p2 = [(166, 378), (178, 375), (197, 374), (218, 382), (200, 385), (181, 385)]
p = [(201, 377), (190, 368), (178, 377), (189, 387)]

def calcular_centroide(pontos):
    x_centro = sum(ponto[0] for ponto in pontos) / len(pontos)
    y_centro = sum(ponto[1] for ponto in pontos) / len(pontos)
    return (int(x_centro), int(y_centro))

# Centroide dos pontos em p
#p1 = calcular_centroide(p)

# Função para calcular as distâncias
def calcular_distancias(pontos, ponto_referencia):
    distancias = []
    for ponto in pontos:
        distancia = math.sqrt((ponto[0] - ponto_referencia[0]) ** 2 + (ponto[1] - ponto_referencia[1]) ** 2)
        distancias.append(int(round(distancia)))  # Arredonda a distância e converte para inteiro
    return distancias


#nao faz sentido [23.021728866442675, 11.180339887498949, 8.54400374531753, 29.427877939124322, 13.601470508735444, 11.313708498984761] a distancia do ponto b ser tao semelhante com a posição ultima
#

# Função para configurar e atualizar o gráfico
def configurar_grafico():
    plt.ion()
    fig, ax = plt.subplots()
    eye_plot, = ax.plot([], [], 'bo', label='Pontos em p2')
    iris_plot, = ax.plot([], [], 'ro', label='Íris')
    centroid_plot, = ax.plot([], [], 'gx', markersize=10, label='Centroide da Íris')
    ax.legend()
    ax.set_xlim(0, 640)  # Ajuste conforme a resolução do vídeo
    ax.set_ylim(0, 480)
    ax.set_title("Posições dos Landmarks em Tempo Real")
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    return fig, ax, eye_plot, iris_plot, centroid_plot


# Lista global para armazenar as anotações dos pontos
anotacoes = []


def atualizar_grafico(eye_plot, iris_plot, centroid_plot, eye_coords, iris_coords, iris_centroid):
    global anotacoes

    # Limpa as anotações anteriores
    for anotacao in anotacoes:
        anotacao.remove()
    anotacoes.clear()

    # Atualiza os pontos de eye_coords e adiciona novas anotações
    if eye_coords:
        eye_plot.set_data(*zip(*eye_coords))
        for i, (x, y) in enumerate(eye_coords):
            anotacao = plt.annotate(f'p{i + 1}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center',
                                    color='blue')
            anotacoes.append(anotacao)  # Armazena a anotação para futura limpeza

    # Atualiza os pontos de iris_coords
    if iris_coords:
        iris_plot.set_data(*zip(*iris_coords))

    # Atualiza o ponto do centroide
    centroid_plot.set_data([iris_centroid[0]], [iris_centroid[1]])

    plt.draw()
    plt.pause(200)  # Pausa para simular vídeo em tempo real


def calcular_porcentagem_contribuicao(distancias, indices):
    """
    Calcula a porcentagem de contribuição de pontos específicos para a soma total das distâncias.

    Parâmetros:
    - distancias (list): Lista de distâncias.
    - indices (list): Lista de índices dos pontos cuja contribuição será calculada.

    Retorna:
    - float: Porcentagem de contribuição dos pontos especificados.
    """
    # Verifica se todos os elementos em distancias são numéricos
    if not all(isinstance(x, (int, float)) for x in distancias):
        raise TypeError("Todos os elementos em 'distancias' devem ser inteiros ou floats.")

    # Soma das distâncias dos pontos especificados pelos índices
    soma_selecionada = sum(distancias[i] for i in indices if isinstance(i, int) and i < len(distancias))

    # Soma total das distâncias
    soma_total = sum(distancias)

    # Cálculo da porcentagem de contribuição
    if soma_total == 0:
        raise ValueError("A soma total das distâncias não pode ser zero para calcular a porcentagem.")
    porcentagem_contribuicao = (soma_selecionada / soma_total) * 100

    return porcentagem_contribuicao

def calcular_porcentagem_contribuicao_3(distancias, indice1, indice2, indice3):
    """
    Calcula a porcentagem de contribuição de três pontos específicos para a soma total das distâncias.

    Parâmetros:
    - distancias (list): Lista de distâncias.
    - indice1, indice2, indice3 (int): Índices dos pontos cuja contribuição será calculada.

    Retorna:
    - float: Porcentagem de contribuição dos três pontos especificados.
    """
    # Verifica se todos os elementos em distancias são numéricos
    if not all(isinstance(x, (int, float)) for x in distancias):
        raise TypeError("Todos os elementos em 'distancias' devem ser inteiros ou floats.")

    # Verifica se os índices são válidos e dentro do limite da lista
    indices = [indice1, indice2, indice3]
    for i in indices:
        if not isinstance(i, int) or i >= len(distancias):
            raise IndexError("Índice fora do alcance da lista 'distancias'.")

    # Soma das distâncias dos pontos especificados pelos três índices
    soma_selecionada = sum(distancias[i] for i in indices)

    # Soma total das distâncias
    soma_total = sum(distancias)

    # Cálculo da porcentagem de contribuição
    if soma_total == 0:
        raise ValueError("A soma total das distâncias não pode ser zero para calcular a porcentagem.")
    porcentagem_contribuicao = (soma_selecionada / soma_total) * 100

    return porcentagem_contribuicao


def calcular_contribuicao_indices(distancias):
    """
    Calcula a contribuição percentual dos índices 0 e 4 em relação à soma total das distâncias.

    Parâmetros:
    - distancias (list): Lista de distâncias entre pontos.

    Retorna:
    - dict: Um dicionário com as contribuições percentuais dos índices 0 e 4.
    """
    # Verifica se há pelo menos cinco elementos na lista
    if len(distancias) < 5:
        raise IndexError("A lista 'distancias' precisa ter pelo menos cinco elementos.")

    # Verifica se todos os elementos são numéricos
    if not all(isinstance(x, (int, float)) for x in distancias):
        raise TypeError("Todos os elementos em 'distancias' devem ser inteiros ou floats.")

    # Soma total das distâncias
    soma_total = sum(distancias)
    if soma_total == 0:
        raise ValueError("A soma total das distâncias não pode ser zero para calcular a porcentagem.")

    # Calcula as contribuições percentuais para os índices 0 e 4
    contribuicao_0 = (distancias[0] / soma_total) * 100
    contribuicao_4 = (distancias[4] / soma_total) * 100
    #indice 0,4
    return contribuicao_0,contribuicao_4


def calcular_contribuicao_indices(distancias, indice1, indice2):
    """
    Calcula a contribuição percentual de dois pontos específicos em relação à soma total das distâncias.

    Parâmetros:
    - distancias (list): Lista de distâncias.
    - indice1, indice2 (int): Índices dos pontos cuja contribuição será calculada.

    Retorna:
    - dict: Dicionário com as contribuições percentuais dos índices especificados.
    """
    # Verifica se os índices estão dentro do alcance da lista
    for i in [indice1, indice2]:
        if i >= len(distancias):
            raise IndexError(f"Índice {i} fora do alcance da lista 'distancias' com comprimento {len(distancias)}.")

    # Soma total das distâncias
    soma_total = sum(distancias)
    if soma_total == 0:
        raise ValueError("A soma total das distâncias não pode ser zero para calcular a porcentagem.")

    # Calcula as contribuições percentuais para os índices especificados
    contribuicao_indice1 = (distancias[indice1] / soma_total) * 100
    contribuicao_indice2 = (distancias[indice2] / soma_total) * 100

    return contribuicao_indice1, contribuicao_indice2


def calcular_direcao(distancias):
    # Grau de Esquerda~Direita
    esquerda = calcular_porcentagem_contribuicao_3(distancias, 0, 1, 5)
    direita = calcular_porcentagem_contribuicao_3(distancias, 2, 4, 3)
    direcao_horizontal = "centro" if abs(esquerda - direita) < 5 else ("esquerda" if esquerda < direita else "direita")

    # Limpa os medidores horizontais e calcula a direção vertical
    distancias_verticais = distancias.copy()
    distancias_verticais.pop(0)
    distancias_verticais.pop(2)
    cima = calcular_porcentagem_contribuicao(distancias_verticais, [3, 2])
    baixo = calcular_porcentagem_contribuicao(distancias_verticais, [1, 2])
    direcao_vertical = "centro" if abs(cima - baixo) < 5 else ("cima" if cima < baixo else "baixo")

    return {
        "direcao_horizontal": direcao_horizontal,
        "direcao_vertical": direcao_vertical,
        "contribuicao_horizontal": {"esquerda": esquerda, "direita": direita},
        "contribuicao_vertical": {"cima": cima, "baixo": baixo}
    }


def calcular_graus(distancias):
    """
    Calcula o grau horizontal e o grau vertical dos pontos com base nas porcentagens de contribuição.

    Parâmetros:
    - distancias (list): Lista de distâncias entre pontos.

    Retorna:
    - dict: Dicionário com o grau horizontal e o grau vertical.
    """
    # Cálculo de Esquerda e Direita (ANTIGO)
    esquerda = calcular_porcentagem_contribuicao_3(distancias, 0, 1, 5)
    direita = calcular_porcentagem_contribuicao_3(distancias, 2, 4, 3)
    grau_horizontal = esquerda - direita

    # Limpa os medidores horizontais para o cálculo vertical
    distancias_verticais = distancias.copy()
    distancias_verticais.pop(0)
    distancias_verticais.pop(2)

    # Define listas de distâncias verticais com base na direção horizontal
    #if grau_horizontal <= 0:
    #   # Cálculo para a tendência na esquerda (P2, P6 - Baixo, Cima)
    #    print("Entrou na esquerda")
    #    distancias_verticais_esquerda = distancias_verticais.copy()
    #    distancias_verticais_esquerda.pop(1)
    #    distancias_verticais_esquerda.pop(1)
    #    baixo, cima = calcular_contribuicao_indices(distancias_verticais_esquerda, 0, 1)
    #else:
    #    # Cálculo para a tendência na direita (P3, P5 - Baixo, Cima)
    #   print("Entrou na Direita")
    #    distancias_verticais_direita = distancias_verticais.copy()
    #    distancias_verticais_direita.pop(0)
    #    distancias_verticais_direita.pop(2)
    #   baixo, cima = calcular_contribuicao_indices(distancias_verticais_direita, 0, 1)
    #    print(baixo,cima)
    # Calcula o grau vertical
    cima = calcular_porcentagem_contribuicao(distancias, [5, 4])
    baixo = calcular_porcentagem_contribuicao(distancias, [1, 2])

    grau_vertical = baixo - cima

    return grau_horizontal,grau_vertical


def calcular_direcao_paraconsistente(distancias):
    """
    Calcula a direção horizontal e vertical com base na lógica paraconsistente.

    Parâmetros:
    - distancias (list): Lista de distâncias entre pontos.

    Retorna:
    - str: Direção com base nos graus horizontal e vertical.
    """
    # Calcula os graus horizontal e vertical
    grau_horizontal,grau_vertical = calcular_graus(distancias)

    # Determina a direção horizontal
    if -8 <= grau_horizontal <= 8:
        direcao_horizontal = "meio"
    elif grau_horizontal > 8:
        direcao_horizontal = "direita"
    else:
        direcao_horizontal = "esquerda"

    #Como o olho vai se comprimindo ao chegar nas bordas é necessaro fazer esse verificação
    valor_ajustado = calcular_valor_ajustado(grau_horizontal)
    #abs(valor_ajustado)
    # Determina a direção vertical
    if -valor_ajustado <= grau_vertical <= valor_ajustado:
        direcao_vertical = "meio"
    elif grau_vertical > valor_ajustado:
        direcao_vertical = "cima"
    else:
        direcao_vertical = "baixo"

    # Análise de direção com base nas combinações de horizontal e vertical
    if direcao_horizontal == "meio" and direcao_vertical == "meio":
        direcao_final = "centro"
    elif direcao_horizontal == "meio":
        direcao_final = direcao_vertical
    elif direcao_vertical == "meio":
        direcao_final = direcao_horizontal
    else:
        direcao_final = f"{direcao_vertical}-{direcao_horizontal}"

    return direcao_final


def calcular_valor_ajustado(valor):
    """
    Calcula um valor ajustado entre 3 e 5 para entradas de -100 a 100,
    mantendo a simetria para valores negativos.

    Parâmetros:
    - valor (float): Um número entre -100 e 100.

    Retorna:
    - float: Valor ajustado entre 3 e 5 (ou -3 e -5).
    """
    # Limita o valor de entrada entre -100 e 100
    valor = max(min(valor, 100), -100)

    # Preserva o sinal do valor original para garantir simetria
    sinal = 1 if valor >= 0 else -1
    valor_abs = abs(valor)

    if valor > 30:
        # Calcula a transição suave entre 3 e 5, aplicando o sinal no final
        resultado = sinal * (5 - (1.25 * (valor_abs / 100) ** 0.24))
        return resultado
    else:
        return 5.0



# Supondo que calcular_centroide(p) retorna as coordenadas do centróide
#p1 = calcular_centroide(p)
p1 = (193,384)
distancias = calcular_distancias(p2, p1)
#print(calcular_direcao_paraconsistente(distancias))
#cima = calcular_porcentagem_contribuicao(distancias, [5, 4])
#baixo = calcular_porcentagem_contribuicao(distancias, [1, 2])
#print(cima,baixo)
#print(baixo-cima)
#valor = baixo-cima
#print("valor ajustado:",calcular_valor_ajustado(-5))

#print(calcular_direcao_paraconsistente(distancias))
# Calcula as distâncias entre p2 e o centróide p1
#print(distancias)

# Calcula a porcentagem de contribuição para descobrir se ele está no meio
#se distancia ente p1 e p4 concentrada for mais de 50% entre os pontos está no centro
p0 = calcular_porcentagem_contribuicao(distancias, [0, 3])
#print(p0)
#print("percentaul de diferença:", calcular_contribuicao_indices(distancias))
#se for mais de 50% ele não tenta nem descobrir se ta indo pra direita ou esquerda e sim vai
#descobrir se estamos indo pra baixo ou pra cima.

#Calculando Porcentagem pra descobrir grau de Esquerda~Direita
#Quanto menor o numero maior o grau de certeza ele é pra um lado.
#print("esquerda",calcular_porcentagem_contribuicao_3(distancias,0,1,5))
#print("direita",calcular_porcentagem_contribuicao_3(distancias,2,4,3))
#se a diferença for menos que 5% está no centro

#distancias2 = distancias.copy()
#-----------------------------------------------------------------------------------#-
#precisamos limpar os medidores de distancia horizontal para ter um calculo mais limpo sobre a distancia vertical.
#print(distancias)
#distancias.pop(0)
#distancias.pop(2)
#print(distancias)
#Calculando Porcentagem pra descobrir grau de Cima~Baixo
#Se a diferença entre ambos for mais que 5% está no centro.

esquerda = calcular_porcentagem_contribuicao_3(distancias, 0, 1, 5)
direita = calcular_porcentagem_contribuicao_3(distancias, 2, 4, 3)
# Limpa os medidores horizontais e calcula a direção vertical
distancias_verticais = distancias.copy()
distancias_verticais.pop(0)
distancias_verticais.pop(2)
#print(distancias_verticais)
#distancias_verticais_direita = distancias_verticais.copy()
#distancias_verticais_esquerda = distancias_verticais.copy()
#distancias_verticais_esquerda.pop(1)
#distancias_verticais_esquerda.pop(1)
#print(distancias_verticais_esquerda)#P2,P6 (BAIXO,CIMA)
#distancias_verticais_direita.pop(0)
#distancias_verticais_direita.pop(2)#P3,P5 (BAIXO,CIMA)
#print(distancias_verticais_direita)


#print(distancias_verticais)
#1,2,3,4,5,6
#2,3,5,6
#if grau_horizontal >= 0:
#vertical_direita = calcular_porcentagem_contribuicao(distancias_verticais_direita, [0, 1])
#vertical_esquerda = calcular_porcentagem_contribuicao(distancias_verticais_esquerda, [0, 1])
#print(vertical_direita)
#print(vertical_esquerda)

#calcular_contribuicao_indices(distancias_verticais_direita, 0, 1))
#print(calcular_contribuicao_indices(distancias_verticais_esquerda, 0, 1))

#cima = calcular_porcentagem_contribuicao(distancias_verticais, [3, 2])
#baixo = calcular_porcentagem_contribuicao(distancias_verticais, [1, 2])


#1,2,3,4,5,6
#2,3,5,6

#se valor negativo, tendendo pra esquerda
#grau_horizontal = esquerda-direita
#if grau_horizontal >= 0:
#    print("Entrou na Esquerda")
#    distancias_verticais_esquerda = distancias_verticais.copy()
#    distancias_verticais_esquerda.pop(1)
#    distancias_verticais_esquerda.pop(1)
#    print(distancias_verticais_esquerda)
#    print(distancias_verticais_esquerda)#P2,P6 (BAIXO,CIMA)
#    print(calcular_contribuicao_indices(distancias_verticais_esquerda, 0, 1))
#    baixo, cima = calcular_contribuicao_indices(distancias_verticais_esquerda, 0, 1)
#    grau_vertical = baixo-cima
    #print(grau_vertical)
#else:
#    print("Entrou na direita")
#    distancias_verticais_direita = distancias_verticais.copy()
#    distancias_verticais_direita.pop(0)
#    distancias_verticais_direita.pop(2)  #P3,P5 (BAIXO,CIMA)
#    print(distancias_verticais_direita)
#    baixo, cima = calcular_contribuicao_indices(distancias_verticais_direita, 0, 1)
   #print(baixo,cima)
#    grau_vertical = baixo-cima

#print(grau_vertical)
grau_horizontal, grau_vertical = calcular_graus(distancias)

#grau_horizontal de -5 a 5 está no meio
#grau horizontal quanto mais vai pro lado negativo, mais pende a esquerda
#grau horizontal quanto mais vai pro lado positivo mais pende a direita

#grau vertical de -6 a 6 está no meio,
#grau vertical quanto mais pro lado positivo mais pra cima está
#grau vertical quanto mais pro lado negativo mais pra baixo está

#negativo é esquerda, positivo é direita. -5 a 5
print("Horizontal: ", grau_horizontal)
#negativo é pra baixo, positivo é pra cima. <6 pra definir cima~baixo
print("Vertical: ", grau_vertical)



# Configura o gráfico e atualiza com os dados necessários
fig, ax, eye_plot, iris_plot, centroid_plot = configurar_grafico()
atualizar_grafico(eye_plot, iris_plot, centroid_plot, p2, p, p1)

# Finalmente, imprime ou utiliza o valor calculado p0 conforme necessário
print("Porcentagem de contribuição de p0 e p3:", p0)

print(calcular_direcao(distancias))
#p1, p2, p6 direita
#p1 horizontal
#p2 vertical baixo
#p6 vertical acima

#p4 horizontal
#p5 vertical acima
#p3 vertical abaixo


#se distancia ente p1 e p4 concentrada for mais de 50% entre os pontos está no centro
#se a distancia de p1 e p4 != 50% então começamos a verificação, se ele está tendendo pra direita ou pra esquerda
#