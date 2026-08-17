# EyeTeractive

Rastreamento ocular assistivo de baixo custo com webcam comum. Traduz a
direção do olhar em comandos, sem hardware dedicado.

Sistema apresentado no **INTERACT 2025** (20th IFIP TC13) e no **IHC 2025**
(*EyeTeractive: Sistema Multiplataforma de Rastreamento Ocular para Interação
Assistiva de Baixo Custo*), Universidade Federal de Rondônia.

## O que o sistema faz — e o que não faz

O EyeTeractive **classifica a direção do olhar em nove categorias discretas**:
centro, cima, baixo, esquerda, direita e as quatro diagonais. É o suficiente
para comandar uma interface — mover um cursor, navegar um menu, acionar uma
seleção.

Ele **não é um estimador de ponto de olhar**. Não produz coordenada na tela,
não segmenta fixações e sacadas, não faz pupilometria. Para estudos de
psicologia experimental que dependem dessas medidas, é o instrumento errado —
a diferença não é de precisão, é de natureza da saída.

## Arquitetura

Cada eixo é decidido pela fonte que consegue medi-lo:

| eixo | quem decide | por quê |
|---|---|---|
| **vertical** (cima/centro/baixo) | **CNN** ResNet-101 | a pálpebra acompanha o olhar vertical e comprime a excursão da íris; a geometria não tem resolução aqui |
| **horizontal** (esquerda/centro/direita) | **geometria** da íris | excursão ampla, interpretável, e roda sem GPU |
| **arbitragem** | **lógica paraconsistente** Eτ | funde as fontes e recusa decidir sob contradição |

As nove direções são o produto cartesiano dos dois eixos ternários.

O eixo horizontal depende apenas dos landmarks do MediaPipe, não do modelo
treinado — o que o torna independente da base usada no treino.

### Nove direções a partir de cinco classes

A rede prevê `center`, `down`, `left`, `right`, `up`. Cada classe é um par
*(horizontal, vertical)* com um dos eixos no centro, então marginalizar o
softmax por eixo dá evidência independente para cada um:

```
P(vertical = cima)   = p_up
P(vertical = baixo)  = p_down
P(vertical = centro) = p_center + p_left + p_right

P(horizontal = esquerda) = p_left
P(horizontal = direita)  = p_right
P(horizontal = centro)   = p_center + p_up + p_down
```

Um olhar diagonal reparte a massa entre `up` e `left`; nas marginais isso vira
evidência simultânea nos dois eixos, e a camada paraconsistente compõe
`cima-esquerda`. Sem classes novas e sem retreino.

### O papel da lógica paraconsistente

Cada eixo vira uma proposição bipolar anotada `(μ, λ)` — para o vertical, *"o
olhar está para cima"*, com `μ` vindo da evidência de cima e `λ` da de baixo.
Delas derivam o **grau de certeza** `Gc = μ − λ`, o **grau de contradição**
`Gct = μ + λ − 1` e o **grau de certeza real** `Gcr`, que desconta a certeza
pela contradição.

A decisão sai de `Gcr`, não de `Gc`. Quando a geometria aponta um lado e a
rede aponta o outro, `Gcr` colapsa e o eixo cai em "centro": o sistema
**recusa** o comando em vez de escolher um dos dois ou oscilar entre eles.

A distinção entre *inconsistência* (`⊤`, fontes conflitando) e
*paracompletude* (`⊥`, ausência de evidência) também é reportada, porque
pedem respostas diferentes — a primeira sugere um quadro ruim, a segunda um
olhar genuinamente centrado.

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/baixar_modelos.sh
```

Python 3.10+. Com GPU NVIDIA a inferência usa CUDA automaticamente; sem ela,
roda em CPU.

Requer **MediaPipe 1.0+** (API Tasks). Os pesos treinados não acompanham o
repositório — ver *Sobre a base de dados*.

## Uso

```bash
# Visualização ao vivo, com as evidências de cada eixo sobrepostas
python scripts/webcam.py --camera 0

# Servidor de inferência (a GPU fica aqui; o cliente só captura e exibe)
python scripts/servidor.py --porta 5000
python scripts/cliente.py --host 192.168.0.10 --fonte 0

# Calibração por usuário
python scripts/calibrar.py --fonte webcam --usuario nome

# Treino
python scripts/treinar.py --dataset data/dataset --epocas 35 --amp --limite-gpu 0.45
```

Como biblioteca:

```python
import cv2
from eyeteractive.pipeline import PipelineOlhar

captura = cv2.VideoCapture(0)
with PipelineOlhar() as pipeline:
    while True:
        ok, quadro = captura.read()
        if not ok:
            break
        analise = pipeline.processar(quadro, timestamp_ms=...)
        if analise.rosto_detectado:
            print(analise.resultado.resumo())
            # cima   conf=0.93 V[cima Gcr=+0.93 verdadeiro] H[centro Gcr=+0.00 indeterminado]
```

Todo script que usa GPU aceita `--limite-gpu`, a fração de VRAM que o processo
pode reservar, para conviver com outros trabalhos na mesma placa.

## Calibração por usuário

A excursão do grau horizontal depende do formato do olho e da distância à
câmera, e o equilíbrio entre sensibilidade e estabilidade depende de quanto o
usuário consegue fixar o olhar. `scripts/calibrar.py` roda uma sessão curta
com cinco alvos e ajusta limiares e escalas por busca em grade — sem retreino
da rede, só a camada de decisão.

```python
from eyeteractive.calibracao import carregar_perfil
from eyeteractive.pipeline import PipelineOlhar

pipeline = PipelineOlhar(config=carregar_perfil("perfis/nome.json"))
```

Sob empate, a calibração escolhe o limiar mais alto. Em uso assistivo um
comando errado custa mais caro que um comando ausente, e o efeito medido é
esse: os erros deixam de ser decisões equivocadas e passam a ser omissões.

## Estrutura

```
src/eyeteractive/
  paraconsistent.py   álgebra Eτ: anotações, Gc, Gct, Gcr, operadores
  geometry.py         centroide da íris, graus por eixo, abertura ocular (EAR)
  cnn.py              ResNet-101 e marginalização do softmax por eixo
  landmarks.py        FaceLandmarker: contorno, íris e recorte do olho
  fusion.py           combina as fontes nas nove direções, com histerese
  calibracao.py       ajuste dos parâmetros de decisão por usuário
  pipeline.py         quadro → direção, de ponta a ponta
  io/                 protocolo de rede e servidor de inferência
scripts/              treino, avaliação, calibração, rotulagem, servidor
tests/                65 testes, sem dependência de GPU ou de pesos
```

## Protocolo de rede

O desenho é *server-side*: o dispositivo cliente captura e exibe, a inferência
fica no servidor.

```
cliente → servidor : uint32 big-endian (tamanho) + bytes JPEG
servidor → cliente : uint32 big-endian (tamanho) + JSON UTF-8
```

A resposta carrega direção, código do comando, confiança e os graus de certeza
e contradição de cada eixo, para o cliente aplicar a própria política de
aceitação. Códigos `0..4` para as direções cardeais, `5..8` para as diagonais;
`--apenas-cardeais` projeta as diagonais de volta em `1..4`.

## Testes

```bash
pytest
```

Cobrem a álgebra paraconsistente, a geometria e a fusão. Não exigem GPU nem os
pesos treinados.

## Sobre a base de dados

**A base não é distribuída, e não pode ser.** As gravações foram feitas com
colaboradores que assinaram termo de autorização de imagem
(`docs/`) autorizando o uso **exclusivamente para treinamento do sistema, no
âmbito acadêmico** — o que não abrange redistribuição. O material original
tampouco existe mais.

Consequências para quem for usar o repositório:

- Os pesos treinados não acompanham o código. `scripts/treinar.py` treina a
  partir de uma base própria, organizada em `data/dataset/{train,val}/<classe>`.
- O eixo **horizontal** funciona sem modelo treinado: depende só dos landmarks
  do MediaPipe. `PipelineOlhar(usar_cnn=False)` já opera nesse modo.

## Escopo da validação

Os resultados publicados vêm de gravações de **um único participante adulto**.
Sob divisão por sessão — treino e validação em gravações distintas — o
desempenho fica em torno de 98% para esse participante.

Isso é coerente com o desenho do sistema, que é **assistivo e personalizado**:
calibra por usuário e opera para uma pessoa por vez. Mas delimita o que se
pode afirmar:

- **A transferência para outra pessoa não foi medida.** Em especial para
  crianças e adolescentes, cuja anatomia ocular e controle oculomotor não são
  extrapoláveis a partir de adultos.
- Não há classe para **olho fechado**; durante uma piscada a rede ainda emite
  uma distribuição sobre direções. A confiança derivada da abertura ocular
  (EAR) atenua isso no eixo vertical, mas não o resolve.
- O sistema **não compensa pose da cabeça**. Movimento de cabeça durante o uso
  degrada a leitura.

## Equipe

- Jáder Louis de Souza Gonçalves — UNIR
- Prof. Dr. Lucas Marques da Cunha — UNIR (orientação)
