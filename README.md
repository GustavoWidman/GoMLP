# GoMLP

## Descrição

Essa atividade tinha como objectivo o desenvolvimento de um Perceptron Multicamadas (MLP) para a solução do problema XOR. Nesse repositório, você encontrará duas implementações do MLP, uma usando Go (e a lib `gonum`) e outra usando Python (e a lib `pytorch`).

## Estrutura

```bash
.
├── go
│   ├── go.mod
│   ├── go.sum
│   └── src
│       ├── main.go
│       ├── math
│       │   └── math.go
│       ├── perceptron
│       │   └── main.go
│       └── utils
│           ├── text.go
│           └── time.go
├── python
│   ├── requirements.txt
│   └── src
│       ├── main.py
│       ├── perceptron
│       │   └── main.py
│       └── utils
│           ├── math.py
│           └── text.py
└── README.md
```

A estrutura do repositório é bem simples. Dentro da pasta `go` você encontrará o código em Go e dentro da pasta `python` você encontrará o código em Python. Dentro de cada uma dessas pastas, você encontrará um arquivo `main.go` ou `main.py` que é o arquivo principal do projeto. Dentro da pasta `perceptron` você encontrará o código do Perceptron Multicamadas (orientado a objetos) e dentro da pasta `utils` você encontrará funções utilitárias e, no caso do Go, também existe uma pasta `math` com funções matemáticas (para compensar a falta de funções matemáticas que existem no `numpy` do Python).

## Execução

### Go

Para executar o código em Go, você deve ter o Go instalado na sua máquina. Para instalar o Go, siga as instruções [aqui](https://golang.org/doc/install). Após instalar o Go, você deve executar o seguinte comando:

```bash
go run src/main.go
```

Você também pode compilar o código e executar o binário gerado:

```bash
go mod tidy # Para garantir que todas as dependências estão instaladas
go build -o go-mlp -ldflags "-s -w" src/main.go # Para compilar o código
./go-mlp # Executar o programa compilado
```

A seguir você encontrará a demonstração do código em Go.



### Python

Para executar o código em Python, você deve ter o Python instalado na sua máquina. Para instalar o Python, siga as instruções [aqui](https://www.python.org/downloads/). Após instalar o Python, você deve executar o seguinte comando:

Crie um ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o código:

```bash
python src/main.py
```

A seguir você encontrará a demonstração do código em Python.



## Resultados

Em Python, o MLP foi treinado em 50 mil épocas e com uma taxa de aprendizado de 0.03, obtendo uma acurácia de 100% (sem arredondamento). Em Go, o MLP foi treinado em 500 mil épocas e com uma taxa de aprendizado de 0.10, obtendo uma acurácia média de 99.35% (sem arredondamento). Com arredondamento, a acurácia de ambos os modelos foi de 100%, o que significa que o modelo foi capaz de simular a função XOR (não aprender, ja que computadores não aprendem). Vale notar que a velocidade de execução do código em Go é muito superior ao código em Python, rodando 10 vezes mais épocas 10 vezes mais rápido (aproximadamente 100 vezes mais rápido).