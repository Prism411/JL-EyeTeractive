import matplotlib.pyplot as plt
import math

# Dados dos pontos
#p2 = [(166, 378), (178, 375), (197, 374), (218, 382), (200, 385), (181, 385)]
#p = [(201, 377), (190, 368), (178, 377), (189, 387)]

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
        distancias.append(distancia)
    return distancias

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
    plt.pause(0.01)  # Pausa para simular vídeo em tempo real
