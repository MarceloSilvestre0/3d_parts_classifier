
![Logo](https://i.pinimg.com/originals/f6/fa/32/f6fa321ae32a49735cb21dcaccdc378b.gif)
# Automatização da Classificação e Análise Espacial de Peças de Xadrez em Modelos 3D STL com Redes Neurais Convolucionais

Automating Classification and Spatial Analysis of Chess Pieces in 3D STL Models with Convolutional Neural Networks


##  Descrição Do Projeto
Este projeto implementa uma Rede Neural Convolucional (CNN) para classificação automática de partes tridimensionais (3D) fornecidas pelo usuário. O sistema engloba um pipeline robusto que inclui a extração, pré-processamento e divisão dos dados, modelagem eficiente, treinamento, e avaliação rigorosa.

Utilizando conceitos avançados de Ciência de Dados e Deep Learning, o projeto visa oferecer uma solução automatizada e precisa para classificação industrial, aplicando técnicas especializadas para manipulação e análise de dados 3D.

------------------------------------------ Pipeline do Projeto -------------------------------------------

![Logo](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/b4fcf973daf0b0cfa2ff77ca8d6816a1/dc937e59-1594-415b-95e9-78e7178b22ca/6d35dc00.png)





![Logo](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/9dd4a3fc74e8a797ae666824056f19d9/a9772cab-9851-4920-9318-eb7a70c59006/4e79d959.png)






## Estrutura do Projeto
* main.py: Orquestra a execução do pipeline, coordenando o fluxo de dados e os processos de treino e inferência.

* classifier.py: Implementa a arquitetura da rede neural convolucional, detalhes do modelo, funções de treinamento e testes.

* data_preparation.py: Contém funções para tratamento e transformação dos dados brutos para o formato adequado ao modelo.

* data_split.py: Responsável pela divisão dos dados em conjuntos de treino, validação e teste, garantindo a validação cruzada correta e evitando vazamentos.

* MathPartProcess.py: Processa especificamente as partes 3D, incluindo leitura, manipulação matemática e extração de características estruturais.

* pipeline_fit.py: Monta o pipeline completo integrando preparação, treino e avaliação dos modelos.

Arquivos de teste (teste_*.py): Garantem a validação de funcionalidades essenciais do código, promovendo confiabilidade e manutenção facilitada.
## Fundamentos Técnicos e Científicos
O projeto apoia-se em bases teóricas sólidas presentes no artigo TCC associado, que detalha:

* Uso de CNN para extrair características relevantes em dados tridimensionais, assegurando alta generalização.


* Processos rigorosos de pré-processamento e balanceamento dos dados, essenciais para melhorar a qualidade das entradas e desempenho do modelo.

* Versionamento do modelo da CNN através do MLFlow

* Desafios da manipulação dos dados 3D e otimizações computacionais para eficiência no treinamento.

* Testes e comparações de diversas arquiteturas de CNN aplicadas ao modelo. Possibilitando a visualização da evolução do modelo construído.

Esse embasamento fortalece a relevância acadêmica e aplicada do projeto, mostrando aplicação efetiva de práticas científicas na indústria de classificação 3D.
## Funcionalidades Principais
* Classificação de partes 3D com modelo CNN customizado.

* Pipeline modularizado cobrindo desde dados brutos até avaliação final.

* Scripts automatizados para dividir e preparar dados.

* Suite de testes para garantir qualidade e integridade do código.

* Embasamento técnico sólido com foco em performance e escalabilidade.
## Tecnologias e Dependências
Python 3.x

Frameworks e libs: TensorFlow ou PyTorch (conforme implementação no classifier.py)

Bibliotecas auxiliares: NumPy, Pandas, Scikit-learn, entre outras para manipulação e análise de dados.
## Como Executar
1. Clone o repositório:
        git clone https://github.com/MarceloSilvestre0/3d_parts_classifier.git

2. Instale as dependências listadas (caso não exista, crie um arquivo requirements.txt baseado nas libs usadas).

3. Execute o script principal:
        python main.py
    
4. Analise os logs e resultados para acompanhamento do processo.
## Possíveis Melhoria e Extensões

* Adição de tratamento de exceções e validação de entradas para robustez.

* Otimização e tunning do modelo CNN para melhorar acurácia.

* Documentação adicional nos módulos para facilitar onboarding.

* Ampliação do conjunto de testes automatizados.

* Exploração do uso de arquiteturas arquiteturas avançadas (ex.: ResNet, EfficientNet).
