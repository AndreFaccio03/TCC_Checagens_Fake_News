# TCC_Checagens_Fake_News

**Trabalho de Conclusão de Curso — Processamento de Linguagem Natural: uma abordagem voltada à análise de checagem de fake news.**

- André Felipe Romero Faccio (FACOM/UFMS) | andre.faccio@ufms.br (Autor)
- Mariana Caravanti de Souza (FACOM/UFMS) | mariana.caravanti@ufms.br (Orientadora)

## Sobre o projeto

Projeto acadêmico que investiga a aplicação de técnicas de Processamento de Linguagem Natural (PLN) e aprendizado de máquina na detecção automática de fake news, utilizando textos de agências brasileiras de checagem de fatos.

A abordagem envolve a extração de informações dos textos de checagem, a geração de representações vetoriais (*embeddings*) e o treinamento de modelos de classificação.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- BGE-M3
- Jupyter Notebook
- Modelos de Linguagem (LLMs)

## Fluxo de execução

Os arquivos devem ser executados na seguinte ordem:

1. `Extrator_Boato_Fatos.py` — Extração de boatos e fatos.
2. `Uniao_Datasets.ipynb` — União dos conjuntos de dados.
3. `Extracao_Embeddings.ipynb` — Extração dos embeddings textuais.
4. `Treinamento_Modelos.ipynb` — Treinamento e avaliação dos modelos de classificação.

## Modelos de classificação

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Multilayer Perceptron (MLP)

## Dados

O conjunto de dados final utilizado no projeto, para treinamento dos classificadores, está disponível no Google Drive:

- [base_final_com_vetores.parquet](https://drive.google.com/file/d/1cOpedo26vS67ltKQ-GJk8bi03DR8IEkr/view?usp=sharing)