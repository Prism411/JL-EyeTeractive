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


# Supondo que calcular_centroide(p) retorna as coordenadas do centróide
p1 = calcular_centroide(p)
p1 = (170,378)
# Calcula as distâncias entre p2 e o centróide p1
distancias = calcular_distancias(p2, p1)
print(distancias)

# Calcula a porcentagem de contribuição para as distâncias, considerando os índices [0, 3]
p0 = calcular_porcentagem_contribuicao(distancias, [0, 3])
print(p0)

# Configura o gráfico e atualiza com os dados necessários
fig, ax, eye_plot, iris_plot, centroid_plot = configurar_grafico()
atualizar_grafico(eye_plot, iris_plot, centroid_plot, p2, p, p1)

# Finalmente, imprime ou utiliza o valor calculado p0 conforme necessário
print("Porcentagem de contribuição de p0 e p3:", p0)

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